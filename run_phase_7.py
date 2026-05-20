"""Phase 7: Evaluation Framework."""
import sys
sys.path.insert(0, 'src')
import logging
logging.basicConfig(level=logging.INFO)

from evaluation.evaluator import ModelEvaluator

print("=" * 60)
print("PHASE 7: EVALUATION FRAMEWORK")
print("=" * 60)

print("\n[1/3] Loading all model metrics...")
evaluator = ModelEvaluator()
evaluator.load_all_metrics()

print("\n[2/3] Computing model rankings...")
evaluator.rank_models()

print("\n[3/3] Generating evaluation report...")
evaluator.generate_report()

print("\n" + "=" * 60)
print("PHASE 7 COMPLETE")
print("=" * 60)
