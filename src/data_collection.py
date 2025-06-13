from utils.simulator import MassSpringSimulator
from utils.file_saver import create_save_path, save_data_and_plot, save_config

# python -m src.data_collection

def collect_data():
    params = {
        "m": 1.0,
        "k": 10.0,
        "c": 1.0,
        "l": 1.0,                 # 0に収束させるようにするため 0.019 にしたけど違う
        "dt": 0.1,
        "T": 30.0,
        "drive_type": "linear",        # exp, const, linear, triangle, step, trapezoid, wavy_decay
        "drive_param": {"A": 1.0, "b": 1.5, "w":10.0, 
                        "v": 1.0, "period": 2.0, "step_time": 15.0, "rise_time": 5.0}
    }

    simulator = MassSpringSimulator(
        m=params["m"], k=params["k"], c=params["c"], l=params["l"],
        dt=params["dt"], T=params["T"]
    )
    t, x, vx, ax, p, v, a = simulator.simulate(
        drive_type=params["drive_type"],
        drive_param=params["drive_param"]
    )

    save_path = create_save_path(base_dir='data')
    save_data_and_plot(save_path, t, x, vx, ax, p, v, a)
    save_config(save_path, params)

    return t, x, vx, ax, p, v, a

if __name__ == "__main__":
    collect_data()
