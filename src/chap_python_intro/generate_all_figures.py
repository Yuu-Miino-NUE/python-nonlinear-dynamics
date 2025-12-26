from henon_attractor import main as henon_attractor_main
from henon_parameter_scan import main as henon_parameter_scan_main
from henon_multiple_trajectories import main as henon_multiple_trajectories_main
from rossler_timeseries import main as rossler_timeseries_main
from rossler_attractor_3d import main as rossler_attractor_3d_main
from rossler_steady_state import main as rossler_steady_state_main
from rossler_parameter_scan import main as rossler_parameter_scan_main


def main():
    henon_attractor_main()
    henon_parameter_scan_main()
    henon_multiple_trajectories_main()
    rossler_timeseries_main()
    rossler_attractor_3d_main()
    rossler_steady_state_main()
    rossler_parameter_scan_main()


if __name__ == "__main__":
    main()
