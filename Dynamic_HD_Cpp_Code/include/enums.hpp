#ifndef ENUMS_HPP_
#define ENUMS_HPP_

  enum height_types { flood_height = 0, connection_height, null_htype};
  enum merge_types {no_merge = 0,
                    connection_merge_as_primary_flood_merge_as_primary = 1,
                    connection_merge_as_primary_flood_merge_as_secondary,
                    connection_merge_as_primary_flood_merge_not_set,
                    connection_merge_as_secondary_flood_merge_as_primary,
                    connection_merge_as_secondary_flood_merge_as_secondary,
                    connection_merge_as_secondary_flood_merge_not_set,
                    connection_merge_not_set_flood_merge_as_primary,
                    connection_merge_not_set_flood_merge_as_secondary,
                    null_mtype};
  enum basic_merge_types {basic_no_merge = 0, merge_as_primary,
                          merge_as_secondary, basic_null_mtype};
  enum redirect_type : bool {local_redirect = true,non_local_redirect = false};

#endif /* ENUMS_HPP_ */
