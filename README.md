project/
├── __init__.py                # Package initializer
├── config.py                  # Configuration settings
├── core/                      # Core utilities and modules
│   ├── __init__.py
│   ├── model_utils.py         # Model loading and embedding generation utilities
│   ├── audio_utils.py         # Audio recording and processing utilities
│   └── evaluation_utils.py    # Evaluation and metric computation functions
├── experiments/               # Experiment scripts
│   ├── __init__.py
│   ├── live_experiment.py     # Script for live experiments
│   └── batch_experiment.py    # Script for batch experiments
└── app.py                    # Main execution script

This project includes both voice verification and asr for verified users using pre-traained models. Custom dataset was created for testing using SRTA employees recordings. to run, make sure all the dependencies are satisfied. All of them are in the dependency.py file
Audio is streamed and not saved, which allows for faster processing. Currently the project works on CPU (mostly).

python -m app
y -> for live inference
