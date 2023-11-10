#include "base/field.hpp"
#include "base/enums.hpp"

using namespace std;

//provides no check that the array on the right hand side is of the same size
template <typename field_type>
bool field<field_type>::operator==(const field<field_type>& rhs) const {
  auto fields_are_equal = true;
  for (auto i = 0; i < _grid->get_total_size();++i){
  fields_are_equal = (array[i] == rhs.get_array()[i]) && fields_are_equal;
  }
  return fields_are_equal;
}

//check if two field are almost equal to within a tolerance
template <typename field_type>
bool field<field_type>::almost_equal(const field<field_type>& rhs, double absolute_tolerance) const {
  auto fields_are_almost_equal = true;
  for (auto i = 0; i < _grid->get_total_size();++i){
  fields_are_almost_equal = ((absolute_tolerance > (array[i] - rhs.get_array()[i])) &&
                ((array[i] - rhs.get_array()[i]) > -1.0*absolute_tolerance) &&
                  fields_are_almost_equal);
  }
  return fields_are_almost_equal;

}

//Print out nlat and nlon and then the underlying data
template <typename field_type>
ostream& operator<< (ostream& out,const field<field_type>& field_object){
  field_object._grid->for_all_with_line_breaks([&](coords* coords, bool end_of_line){
    int width = 10;
    if (is_same<field_type,int>::value) width = 3;
    if (is_same<field_type,bool>::value) width = 2;
    if (end_of_line) out << endl;
    out <<  setw(width) << setprecision(2) << field_object(coords);
  });
  return out;
}

template class field<short>;
template class field<int>;
template class field<double>;
template class field<bool>;
template class field<channel_type>;
template class field<height_types>;
template ostream& operator<< <int>(ostream& out,const field<int>& field_object);
