/*
 * Class Neurons
 * represents the state (activation level) of a set of neurons
 * in host and device memory
 */

#include "neurons.h"

namespace thinkerbell {

Neurons::Neurons( uint n ) : m_size(n) { }

Neurons::~Neurons() { }

uint Neurons::size() const { return m_size; }

}
