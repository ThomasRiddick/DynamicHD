#ifndef ENUMS_HPP_
#define ENUMS_HPP_

  enum height_types { flood_height = 0, connection_height, null_htype};

  enum geometry_types {icosohedral = 0};

  enum channel_type {not_main_channel = 0,
                     main_channel_valid,
                     main_channel_invalid};

  enum pixel_type { flow_normal_outlet = 0,
                    flow_rmouth_outlet,
                    flow_sink_outlet,
                    flow_source_cell_outlet };

#endif /* ENUMS_HPP_ */
