"""Training and evaluation subsystem for Mangaba AI v3.0.

Two entry points, both reachable from the CLI:

* :class:`~mangaba.training.evaluator.CrewEvaluator` (``mangaba test``) —
  runs a crew N times and scores every task output 1–10 with an LLM judge.
* :class:`~mangaba.training.trainer.CrewTrainer` (``mangaba train``) —
  human-in-the-loop refinement whose result is persisted to
  ``trained_agents_data.pkl``.

Saved training data is re-applied with
:func:`~mangaba.training.trainer.apply_training_data`, which writes a
delimited block into each agent's backstory *and* sets
``agent.training_context``. The preferred long-term hook is for
``Agent.__init__`` to accept ``training_context`` and for
``Agent._build_system_prompt`` to append ``self.training_context`` when set —
with that hook in place the backstory would no longer be mutated.

Example::

    from mangaba.training import CrewEvaluator, CrewTrainer, apply_training_data

    apply_training_data(crew)
    result = CrewEvaluator(crew, iterations=2).evaluate()
"""

from mangaba.training.evaluator import (
    CrewEvaluator,
    EvaluationResult,
    RunScore,
    TaskScore,
)
from mangaba.training.trainer import (
    DEFAULT_TRAINING_FILE,
    AgentTrainingData,
    CrewTrainer,
    TrainingResult,
    apply_training_data,
    load_training_data,
    save_training_data,
    set_training_context,
)

__all__ = [
    # Evaluation
    "CrewEvaluator",
    "EvaluationResult",
    "RunScore",
    "TaskScore",
    # Training
    "CrewTrainer",
    "TrainingResult",
    "AgentTrainingData",
    "DEFAULT_TRAINING_FILE",
    "apply_training_data",
    "load_training_data",
    "save_training_data",
    "set_training_context",
]
