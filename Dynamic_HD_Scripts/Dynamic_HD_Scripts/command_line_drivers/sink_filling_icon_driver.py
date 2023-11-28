// string UNITS = "units";
// string GRID_TYPE = "grid_type";
// string LONG_NAME = "long_name";
// string COORDINATES = "coordinates";
// string STANDARD_NAME = "standard_name";
// string BOUNDS = "bounds";

// string METRES = "m";
// string RADIAN = "radian";
// string UNSTRUCTURED = "unstructured";
// string LATITUDE = "latitude";
// string CENTER_LATITUDE = "center latitude";
// string LONGITUDE = "longitude";
// string CENTER_LONGITUDE = "center longitude";

// void print_usage(){
//     cout <<
//     "Usage: " << endl;
//     cout <<
//     "./Fill_Sinks_Icon_SI_Exec [input orography file path] [landsea file path]" << endl;
//     cout <<
//     " [true sinks file path] [output orography file path] [grid parameters file path]" << endl;
//     cout <<
//     " [input orography field name] [input landsea mask fieldname]" << endl;
//     cout <<
//     " [input true sinks field name] [set land sea as no data flag] [add slope flag]" << endl;
//     cout <<
//     "[epsilon] [use secondary neighbors flag] [fractional landsea mask flag]" << endl;
// }

// void print_help(){
//     print_usage();
//     cout << "Fill the sinks in an orography using an accelerated" << endl;
//     cout << "priority flood technique on a ICON icosohedral grid" << endl;
//     cout << "Arguments:" << endl;
//     cout << "input orography file path - full path to input orography file" << endl;
//     cout << "landsea file path - full path to land-sea input file" << endl;
//     cout << "true sinks file path - full path to true sink input file" << endl;
//     cout << "output orography file path - full path to target output orography file" << endl;
//     cout << "grid parameters file path - full path to file containing ICON grid parameters"
//          << endl << " for resolution being used" << endl;
//     cout << "input orography field name - name of input orography field within"
//          << " specified file" << endl;
//     cout << "input landsea mask field name - name of landsea mask field within"
//          << " specified file" << endl;
//     cout << "input true sinks field name - name of true sinks field within"
//          << " specified file" << endl;
//     cout << "set land sea as no data flag - flag to turn on and off setting"
//          << " all landsea points to no data" << endl;
//     cout << "add slope flag - flag to turn on and off adding a slight slope"
//          << " when filling sinks" << endl;
//     cout << "epsilon - additional height added to each progressive cell when"
//          << " adding a slight slope" << endl;
//     cout << "use secondary neighbors flag - Use the 9 additional neighbors which"
//          << "share vertices"
//          << endl << "but not edges with a cell (default: true)" << endl;
//     cout << "fractional landsea mask - land sea mask expresses fraction of land"
//          << "as a floating point number" << endl
//          << "which requires conversion to a binary mask" << endl;
// }

// int main(int argc, char *argv[]){
//   cout << "ICON sink filling tool" << endl;
//   int opts;
//   while ((opts = getopt(argc,argv,"h")) != -1){
//     if (opts == 'h'){
//       print_help();
//       exit(EXIT_FAILURE);
//     }
//   }
//   if(argc<12) {
//     cout << "Not enough arguments" << endl;
//     print_usage();
//     cout << "Run with option -h for help" << endl;
//     exit(EXIT_FAILURE);
//   }
//   if(argc>14) {
//     cout << "Too many arguments" << endl;
//     print_usage();
//     cout << "Run with option -h for help" << endl;
//     exit(EXIT_FAILURE);
//   }
  // string orography_in_filepath(argv[1]);
  // string landsea_in_filepath(argv[2]);
  // string true_sinks_in_filepath(argv[3]);
  // string orography_out_filepath(argv[4]);
  // string grid_params_filepath(argv[5]);
  // string orography_in_fieldname(argv[6]);
  // string landsea_in_fieldname(argv[7]);
  // string true_sinks_in_fieldname(argv[8]);
  // string set_ls_as_no_data_flag_string(argv[9]);
  // string add_slope_string(argv[10]);
  // string epsilon_string(argv[11]);
  // bool fractional_landsea_mask_in = false;
  // bool use_secondary_neighbors_in;
  // if (argc == 13 || argc == 14) {
  //   string use_secondary_neighbors_string(argv[12]);
  //   use_secondary_neighbors_in = bool(stoi(use_secondary_neighbors_string));
  //   if (argc == 14) {
  //   string fractional_landsea_mask_string(argv[13]);
  //   fractional_landsea_mask_in = bool(stoi(fractional_landsea_mask_string));
  //   }
  // } else use_secondary_neighbors_in = true;
  // ifstream ofile(orography_out_filepath.c_str());
  // if (ofile) {
  //   cout << "Outfile already exists - please delete or specify a different name" << endl;
  //   exit(1);
  // }
  // bool set_ls_as_no_data_flag = bool(stoi(set_ls_as_no_data_flag_string));
  // bool add_slope_in = bool(stoi(add_slope_string));
  // double epsilon_in = stod(epsilon_string);
  //
