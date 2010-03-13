#ifndef __LIST_BOX_GENERIC_CONTAINER_H__
#define __LIST_BOX_GENERIC_CONTAINER_H__

#include <wx/listbox.h>
#include <vector>
#include <thinkerbell/deep_belief_network.h>


template< class T >
class wxListBoxGenericContainer : public wxListBox
{
public:

  wxListBoxGenericContainer( wxWindow* parent
                           , wxWindowID id = -1
                           , const wxPoint& pos = wxDefaultPosition
                           , const wxSize& size = wxDefaultSize
                           , long style=0
                           , int k = NULL
                           , int p = 0
                           )
    : wxListBox( parent, id, pos, size, style, NULL, 0, wxDefaultValidator, _("") )
  {
  }

  void Append(const wxString& item, const T &clientData)
  {
    m_container.push_back(clientData);
    wxListBox::Append(item, (void *)&m_container.back() );
  }

  void Clear()
  {
    m_container.clear();
    wxListBox::Clear();
  }

  T * GetClientData(unsigned int n) const
  {
    return (T *)wxListBox::GetClientData(n);
  }

private:
  std::list<T> m_container;
  
};

typedef wxListBoxGenericContainer< thinkerbell::Vertex > wxListBoxVertices;
typedef wxListBoxGenericContainer< thinkerbell::Edge > wxListBoxEdges;

#endif
