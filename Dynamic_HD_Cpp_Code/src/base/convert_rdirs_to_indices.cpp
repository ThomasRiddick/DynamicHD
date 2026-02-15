#include "base/convert_rdirs_to_indices.hpp"
#include <stdexcept>
using namespace std;

void convert_rdirs_to_latlon_indices(int nlat, int nlon,
                                     int* rdirs,
                                     int* next_cell_index_lat,
                                     int* next_cell_index_lon) {
  for (int i = 0; i < nlat; i++) {
    for (int j = 0; j < nlon; j++) {
      int rdir = rdirs[i*nlon+j];
      if (rdir == 5) {
        next_cell_index_lat[i*nlon+j] = -5;
        next_cell_index_lon[i*nlon+j] = -5;
      } else if (rdir == 0 || rdir == -1 || rdir == -2) {
        next_cell_index_lat[i*nlon+j] = rdir;
        next_cell_index_lon[i*nlon+j] = rdir;
      } else if (rdir <= 9 && rdir >= 1) {
        if (rdir == 7 || rdir == 8 || rdir == 9) {
          if (i == 0) next_cell_index_lat[j] = 0;
          else next_cell_index_lat[i*nlon+j] = i - 1;
        } else if (rdir == 4 || rdir == 6) {
          next_cell_index_lat[i*nlon+j] = i;
        } else if (rdir == 1 || rdir == 2 || rdir == 3) {
          if (i == (nlat -1)) next_cell_index_lat[i*nlon+j] = nlat -1;
          else next_cell_index_lat[i*nlon+j] = i + 1;
        }
        if (rdir == 7 || rdir == 4 || rdir == 1) {
          if (j == 0) next_cell_index_lon[i*nlon] = nlon - 1;
          else next_cell_index_lon[i*nlon+j] = j - 1;
        } else if (rdir == 8 || rdir == 2) {
          next_cell_index_lon[i*nlon+j] = j;
        } else if (rdir == 9 || rdir == 6 || rdir == 3) {
          if (j == (nlon -1)) next_cell_index_lon[i*nlon+j] = 0;
          else next_cell_index_lon[i*nlon+j] = j + 1;
        }
      } else throw runtime_error("Invalid river direction");
    }
  }
}
