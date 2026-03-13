# experiments/smoke_test_combinations.py
import sys
import tensorflow as tf
from src.models import make_dense_model, make_ncp_model

NEURONS = ['lrc', 'lrc_ar', 'ctrnn', 'lstm']
WIRINGS = ['dense', 'ncp']

# Dense config per neuron.
# lrc_ar uses raw input as the ODE state (v_pre = inputs), so mu/sigma weight
# matrices are (units×units). The broadcast in _sigmoid requires input_dim==units.
# For dense wiring with spiral data (features=2), lrc_ar must have units=2.
DENSE_UNITS = {
    'lrc':    8,
    'lrc_ar': 2,
    'ctrnn':  8,
    'lstm':   8,
}

# NCP config — same for all neurons.
# lrc_ar + NCP fails: the inter layer receives features=2 (raw input) but
# inter_neurons=8, so the (units,units)=(8,8) mu/sigma matrices cause a shape
# error in _sigmoid. This is an expected, documented incompatibility.
NCP_CONFIG = dict(inter_neurons=8, command_neurons=6, motor_neurons=2)

# Known-incompatible pairs. These are expected to fail — not bugs.
EXPECTED_FAIL = {('lrc_ar', 'ncp')}


def run_combination(neuron: str, wiring: str, n_iters: int = 3) -> tuple[bool, Exception | None]:
    """Build a model for (neuron, wiring) and run n_iters gradient steps.

    Uses synthetic random data — no real ODE dataset needed.

    Returns:
        (success, error) where error is None on success.
    """
    try:
        x = tf.random.normal((2, 10, 2))
        y = tf.random.normal((2, 10, 2))

        if wiring == 'dense':
            model = make_dense_model(neuron, units=DENSE_UNITS[neuron], output_neurons=2)
        else:
            model = make_ncp_model(neuron, **NCP_CONFIG)

        optimizer = tf.keras.optimizers.Adam()
        for _ in range(n_iters):
            with tf.GradientTape() as tape:
                pred = model(x, training=True)
                loss = tf.reduce_mean(tf.square(pred - y))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return True, None
    except Exception as e:
        return False, e


def main() -> int:
    """Run all combinations. Return exit code (0=success, 1=failure)."""
    results = {}
    for neuron in NEURONS:
        for wiring in WIRINGS:
            ok, err = run_combination(neuron, wiring)
            results[(neuron, wiring)] = (ok, err)

    # Print summary table
    print('\n=== Smoke Test: Cell × Wiring Combinations ===\n')
    print(f'  {"Neuron":<8} {"Wiring":<8} Status')
    print(f'  {"-------":<8} {"-------":<8} ------')

    n_pass = 0
    n_xfail = 0
    exit_code = 0

    for neuron in NEURONS:
        for wiring in WIRINGS:
            ok, err = results[(neuron, wiring)]
            key = (neuron, wiring)
            is_xfail = key in EXPECTED_FAIL

            if is_xfail:
                if ok:
                    # Expected to fail but passed — unexpected success
                    status = '❌ XPASS (unexpected pass — should have failed)'
                    exit_code = 1
                else:
                    status = '⚠️  XFAIL (expected — lrc_ar input constraint)'
                    n_xfail += 1
            else:
                if ok:
                    status = '✅ PASS'
                    n_pass += 1
                else:
                    status = f'❌ FAIL — {type(err).__name__}: {err}'
                    exit_code = 1

            print(f'  {neuron:<8} {wiring:<8} {status}')

    print(f'\nResult: {n_pass} passed, {n_xfail} expected failure(s). '
          f'Exit {exit_code}.')

    return exit_code


if __name__ == '__main__':
    sys.exit(main())
