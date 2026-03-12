import argparse
import yaml
from src.tasks.neural_ode.datasets import generate_dataset
from src.tasks.neural_ode.ode_model import ODEFuncModel
from src.tasks.neural_ode.trainer import train


def main():
    parser = argparse.ArgumentParser(description='Train Neural ODE model')
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Training Neural ODE | data={config['data']} neuron={config['neuron']} units={config['units']}")

    t, y = generate_dataset(config['data'])
    model = ODEFuncModel(config['neuron'], config['wiring'], config['units'], features=2)

    losses = train(
        model, t, y,
        n_iters=config['niters'],
        batch_size=config['batch_size'],
        batch_time=config['batch_time'],
        lr=config['lr'],
    )

    print(f'Training complete. Final loss: {losses[-1]:.6f}')


if __name__ == '__main__':
    main()
