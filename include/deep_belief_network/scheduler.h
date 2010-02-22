#ifndef __DEEP_BELIEF_NETWORK_SCHEDULER_H__
#define __DEEP_BELIEF_NETWORK_SCHEDULER_H__

#include "deep_belief_network.h"
#include <boost/thread/condition.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/xtime.hpp>
#include <vector>
#include <list>
#include <iostream>
#include <cudamm/cuda.hpp>
#include "tmp.h"
#define WITH_LOGGING
#include "logger.h"

namespace thinkerbell {

using namespace cuda;

class DeepBeliefNetworkScheduler
{
  unsigned int num_streams;
  unsigned int batch_size;
  DeepBeliefNetwork * dbn;
private:
  DeepBeliefNetworkScheduler() {} // not default constructible
public:
  explicit
  DeepBeliefNetworkScheduler( DeepBeliefNetwork * dbn_, unsigned int num_streams_, unsigned int batch_size_ )
    : num_streams( num_streams_ ),
      batch_size( batch_size_ ),
      dbn(dbn_)
    {}

  void operator()()
  {
    Logger::log("Scheduler starting...");
    Cuda context(0);
    Module module_test_kernels("src/test_kernels.cubin");
    Function mmul( module_test_kernels, "mmul" );
    Function mmultb( module_test_kernels, "mmul_transpose_b" );
    //Function madd( module_test_kernels, "madd" );
    vector<Stream *> streams;

    map<Vertex, DevicePtr> vertex_ptr;
    map<Edge, DevicePtr> edge_ptr;

    list<Vertex> vertices = dbn->topological_order();

    // allocate device memory for each vertex's neuron values for the up-activation phase (one block per batch per vertex)
    int size_accum = 0;
    typedef list< Vertex >::iterator Viterator;
    for(Viterator vi = vertices.begin(); vi != vertices.end(); vi++ )
    {
      int width = dbn->m_graph[*vi].neurons->size();
      int height = batch_size;
      size_accum += ( width * height );
    }
    DeviceMemory vertex_memory( sizeof(float) * size_accum * num_streams );

    // now go back and store DevicePtrs for each vertex
    DevicePtr current_vertex_ptr = vertex_memory.ptr();
    for(Viterator vi = vertices.begin(); vi != vertices.end(); vi++ )
    {
      int width = dbn->m_graph[*vi].neurons->size();
      int height = batch_size;
      vertex_ptr[*vi] = current_vertex_ptr;
      current_vertex_ptr = current_vertex_ptr + (sizeof(float) * (num_streams * width * height));
    }

    // allocate device memory for each edge's weights (one block per vertex, this memory is not written to so all batches can share it)
    // remember the last edge
    size_accum = 0;
    int last_edge_width, last_edge_height;
    Edge last_edge;
    for(Viterator vi = vertices.begin(); vi != vertices.end(); vi++ )
    {
      // find the out edge for this vertex:
      graph_traits< Graph >::out_edge_iterator out_i, out_end;
      tie( out_i, out_end ) = out_edges( *vi, dbn->m_graph );
      if((out_end - out_i) == 1) // if there's one out edge
      {
        Edge e = last_edge = *out_i;
        Viterator nvi = vi; nvi++;
        int width  = last_edge_width  = dbn->m_graph[*nvi].neurons->size(); // size of the next vertex up
        int height = last_edge_height = dbn->m_graph[*vi].neurons->size();     // size of this vertex
        size_accum += ( width * height );
      }
    }
    DeviceMemory edge_memory( sizeof(float) * size_accum );

    // now go back and store DevicePtrs for each edge
    DevicePtr current_edge_ptr = edge_memory.ptr();
    for(Viterator vi = vertices.begin(); vi != vertices.end(); vi++ )
    {
      // find the out edge for this vertex:
      graph_traits< Graph >::out_edge_iterator out_i, out_end;
      tie( out_i, out_end ) = out_edges( *vi, dbn->m_graph );
      if((out_end - out_i) == 1) // if there's one out edge
      {
        Edge e = last_edge = *out_i;
        Viterator nvi = vi; nvi++;
        int width  = dbn->m_graph[*nvi].neurons->size(); // size of the next vertex up
        int height = dbn->m_graph[*vi].neurons->size();     // size of this vertex
        edge_ptr[e] = current_edge_ptr;
        current_edge_ptr = current_edge_ptr + (sizeof(float) * (width * height));
      }
    }

    // allocate memory for 2 alternating Gibbs sampling phases (a down-activation and an up-activation)
    // the operations are:
    //   a-batch = b-batch * weights(transposed)     <- down activate
    //   b-batch = a-batch * weights                 <- up activate
    // find the two top-level vertices and their dimensions
    Vertex inv = source( last_edge, dbn->m_graph );
    Vertex outv = target( last_edge, dbn->m_graph );
    int inv_width   = dbn->m_graph[inv].neurons->size();
    int inv_height  = batch_size;
    int outv_width  = dbn->m_graph[outv].neurons->size();
    int outv_height = batch_size;

    // each batch gets space for a block of neuron values for inv and outv
    DeviceMemory ags_memory(
      sizeof(float) * num_streams *
      ( inv_width  * inv_height + outv_width * outv_height ) );

    // allocate memory for weight adjustment intermediate values (a space for each batch, only for the last edge)
    DeviceMemory weight_scratch( sizeof(float) * num_streams * last_edge_width * last_edge_height );

    Logger::log("launching kernels...");
    // start launching kernels for the up-activation through the whole graph
    for( int si = 0; si < num_streams; ++si )
    {
      streams.push_back(new Stream());
      // for each vertex
      for(Viterator vi = vertices.begin(); vi != vertices.end(); vi++ )
      {
        // find the out edge for this vertex:
        graph_traits< Graph >::out_edge_iterator out_i, out_end;
        tie( out_i, out_end ) = out_edges( *vi, dbn->m_graph );
        if((out_end - out_i) == 1) // if there's one out edge
        {
          Edge e = last_edge = *out_i;
          Viterator nvi = vi; nvi++;
          int abatch_width = dbn->m_graph[*vi].neurons->size();     // size of this vertex
          int bbatch_width = dbn->m_graph[*nvi].neurons->size(); // size of the next vertex up
          int abatch_height = batch_size;
          int bbatch_height = batch_size;
          int weights_width = bbatch_width;
          int weights_height = abatch_width;
          DevicePtr bbatch  = vertex_ptr[*nvi] + ( sizeof(float) * si * bbatch_width * bbatch_height);
          DevicePtr abatch  = vertex_ptr[*vi] +  ( sizeof(float) * si * abatch_width * abatch_height);
          DevicePtr weights = edge_ptr[e];
          mmul.setBlockShape( BLOCK_SIZE, BLOCK_SIZE, 1 );
          // up activate:
          // b-batch = a-batch * weights
          mmul.go(
            bbatch_width / BLOCK_SIZE,
            bbatch_height / BLOCK_SIZE,
            *streams.back(),
            bbatch,
            abatch,
            weights,
            abatch_width,
            weights_width
          );
        }
      }
    
    }

    Logger::log("waiting on streams...");
    // wait for all to finish
    for( int si = 0; si < num_streams; ++si )
    {
      streams[si]->synchronize();
      Logger::log("stream done...");
      delete streams[si];
    }

  }

};

} // end namespace thinkerbell


#endif
