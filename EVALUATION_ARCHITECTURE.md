# Evaluation Framework Architecture

## Overview

The evaluation framework consists of three main modules that work together to provide comprehensive evaluation of attack and defense mechanisms:

```
┌─────────────────────────────────────────────────────────────────┐
│         run_comprehensive_evaluation.py (Master Script)         │
│                                                                 │
│  Orchestrates all evaluations, handles scheduling and          │
│  aggregates results across all configurations                  │
└────────────┬────────────────────────────────────────────────────┘
             │
             ├─────────────────────┬──────────────────────┬─────────────────────┐
             │                     │                      │                     │
             ▼                     ▼                      ▼                     ▼
┌──────────────────────┐ ┌────────────────────┐ ┌──────────────────┐ ┌────────────────┐
│ evaluation_method.py │ │evaluation_multi_   │ │evaluation_method │ │ auxiliary_eval │
│                      │ │mode.py             │ │_with_defense.py  │ │ modules        │
│ STANDARD MODE        │ │                    │ │                  │ │                │
│ • Direct attacks     │ │MULTI-MODE          │ │DEFENSE MODE      │ │• Metrics      │
│ • No defense         │ │ • Repeated prompt  │ │ • Defense check  │ │• Aggregation  │
│ • 10x sampling       │ │ • Multi-turn       │ │ • Attack check   │ │• Reporting    │
│ • ROUGE+WES metrics  │ │ • Delta tracking   │ │ • Defense+Attack │ │                │
└──────────────────────┘ │ • First/Second     │ │                  │ └────────────────┘
                         │   metrics          │ └──────────────────┘
                         └────────────────────┘
```

## Module Details

### 1. `evaluation_method.py` - Standard Evaluation

**Purpose**: Evaluate attack methods in a straightforward manner

**Key Classes**:
- `EvaluationFramework`: Main evaluation orchestrator
- `EvaluationExample`: Single example result
- `EvaluationMetrics`: Aggregated metrics

**Process Flow**:
```
Load Training Prompts
        ↓
Select Top K Prompts
        ↓
Load Test System Prompts
        ↓
For each Test Prompt:
  For each Top Attack Prompt:
    Query Model 10 Times
    Calculate Similarity Metrics
    Store Example
        ↓
Aggregate Metrics
        ↓
Save Results (metrics, top 5, examples)
```

**Output Files**:
- `metrics_{method}_{model}_{timestamp}.json` - Aggregated metrics
- `top5_examples_{method}_{model}_{timestamp}.json` - Top 5 examples
- `examples_{method}_{model}_{timestamp}.jsonl` - All examples

### 2. `evaluation_multi_mode.py` - Multi-Mode Evaluation

**Purpose**: Evaluate attacks in repeated_prompt and multi_turn scenarios

**Key Classes**:
- `MultiModeEvaluationFramework`: Extended framework for multi-mode
- `MultiModeEvaluationExample`: Example with first/second metrics
- `MultiModeEvaluationMetrics`: Metrics with delta calculations

**Repeated Prompt Mode**:
```
Load Training Prompts
        ↓
Select Top K Prompts
        ↓
Load Test System Prompts
        ↓
For each Test Prompt:
  For each Top Attack Prompt:
    First Turn:
      Query Model 10 Times
      Calculate Metrics
    Second Turn:
      Query Model 10 Times (same prompt)
      Calculate Metrics
      Calculate Delta
        ↓
Aggregate First/Second/Delta Metrics
        ↓
Save Results with Turn Separation
```

**Multi-Turn Mode**:
```
Load Training Prompts
        ↓
Select Top K Prompts (assume 2+ different prompts available)
        ↓
Load Test System Prompts
        ↓
For each Test Prompt:
  For each Pair of Attack Prompts:
    Turn 1:
      Query with Attack Prompt 1 (10 times)
      Calculate Metrics
    Turn 2:
      Query with Attack Prompt 2 (10 times)
      Calculate Metrics
      Calculate Delta
        ↓
Aggregate First/Second/Delta Metrics
        ↓
Save Results with Turn Separation
```

**Output Files**:
- `metrics_{method}_{model}_{timestamp}.json` - First/Second/Delta metrics
- `top5_examples_{method}_{model}_{timestamp}.json` - Top 5 examples (both turns)
- `examples_{method}_{model}_{timestamp}.jsonl` - All examples

### 3. `evaluation_method_with_defense.py` - Defense Evaluation

**Purpose**: Evaluate attack effectiveness against defended models

**Key Classes**:
- `DefendedEvaluationFramework`: Framework with defense integration
- Defense-specific methods: `_apply_promptguard_defense()`, etc.

**Process Flow**:
```
Load Training Prompts
        ↓
Select Top K Prompts
        ↓
Load Test System Prompts
        ↓
Initialize Defense Model
        ↓
For each Test Prompt:
  For each Top Attack Prompt:
    Apply Defense to Attack Prompt
    Query Model 10 Times
    Calculate Similarity Metrics
    Store Example
        ↓
Aggregate Metrics
        ↓
Save Results (with defense label)
```

**Defense Integration**:
```
Attack Prompt → Defense Filter → [Blocked/Modified/Passed] → Model Query
                                            ↓
                                   Similarity Calculation
```

**Output Files**:
- `metrics_{method}_vs_{defense}_{model}_{timestamp}.json` - Aggregated metrics
- `top5_examples_{method}_vs_{defense}_{model}_{timestamp}.json` - Top 5 examples
- `examples_{method}_vs_{defense}_{model}_{timestamp}.jsonl` - All examples

### 4. `run_comprehensive_evaluation.py` - Master Orchestrator

**Purpose**: Coordinate all evaluations and manage workflow

**Key Classes**:
- `ComprehensiveEvaluationRunner`: Master evaluation orchestrator

**Execution Modes**:

1. **Comprehensive Mode** (`--all`):
   - Runs all attack methods × all models × all modes × all defenses
   - Complete coverage of evaluation space

2. **Selective Mode** (`--method` and `--model`):
   - Runs specific method/model combination
   - Allows choosing which modes and defenses to include

3. **Interactive Mode** (`--interactive`):
   - User selects methods, models, modes, and defenses
   - Flexible configuration without command-line arguments

**Workflow Phases**:
```
Phase 1: Standard Evaluations
  For each method × model:
    run evaluation_method.py

Phase 2: Multi-Mode Evaluations
  For each method × model:
    run evaluation_multi_mode.py (repeated_prompt)
    run evaluation_multi_mode.py (multi_turn)

Phase 3: Defense Evaluations
  For each method × model × defense:
    run evaluation_method_with_defense.py

Results Aggregation:
  Collect all metrics
  Generate comparison tables
  Produce summary report
```

## Metrics Hierarchy

```
Raw Metrics (Per Response)
├─ ROUGE-L
├─ WES (Word Edit Similarity)
├─ LCS (Levenshtein Distance)
└─ Highest Similarity (max of above)

Example-Level Metrics
├─ System Prompt
├─ Attack Prompt
├─ 10 Responses
├─ Similarity Scores (all metrics)
└─ Highest Similarity

Aggregated Metrics (Multi-Mode)
├─ First Turn
│  ├─ Average Similarity
│  └─ Max Similarity
├─ Second Turn
│  ├─ Average Similarity
│  └─ Max Similarity
└─ Delta
   ├─ Average Delta
   └─ Max Delta

Aggregated Metrics (Standard)
├─ Average Similarity
├─ Max Similarity
├─ Runtime
└─ Total Queries
```

## Data Flow

### Input Data
```
Training Prompts CSV (from attack generation)
├─ text: adversarial prompt
├─ reward: reward score
└─ other metadata

Test System Prompts CSV
├─ text: system prompt
└─ other metadata
```

### Processing
```
Training Prompts → Select Top K by Reward
                          ↓
Test System Prompts + Top K Attacks → Evaluation Framework
                          ↓
Model Queries (via OpenAI-compatible API)
                          ↓
Response Collection & Metric Calculation
```

### Output Data
```
Metrics JSON
├─ method, model, mode, defense
├─ total_samples
├─ avg/max similarity metrics
├─ runtime, num_queries

Top 5 Examples JSON
└─ Array of:
   ├─ system_prompt
   ├─ attack_prompt
   ├─ responses (list)
   └─ similarity_scores

Examples JSONL
└─ One line per example:
   ├─ system_prompt
   ├─ attack_prompt
   ├─ num_responses
   ├─ similarity_scores
   └─ timestamp
```

## Extensibility Points

### Adding New Metrics
1. Implement in `rewards/text_rewards.py`
2. Call from `calculate_similarity_metrics()`
3. Include in output JSON

### Adding New Defenses
1. Implement `_apply_{defense_name}_defense()` in `DefendedEvaluationFramework`
2. Add to `DEFENSES` list in `run_comprehensive_evaluation.py`
3. Update defense initialization in `_load_defense()`

### Adding New Evaluation Modes
1. Implement new evaluation class (e.g., `ConversationalEvaluationFramework`)
2. Implement mode-specific query/metric methods
3. Create new module (e.g., `evaluation_conversational.py`)
4. Add to master orchestrator

### Adding New Models
1. Add to `MODELS` dict in `run_comprehensive_evaluation.py`
2. Add to model selection in interactive mode
3. Ensure server supports the model

## Performance Characteristics

### Computation
- Standard Mode: N_test × N_top_k × N_samples queries
  - Example: 50 test × 5 prompts × 10 samples = 2,500 queries
  - Time: ~30-60 minutes for Llama 3.1-8B

- Multi-Mode: N_test × N_top_k × N_samples × 2 queries (two turns)
  - Time: ~60-120 minutes

- Defense Mode: Same as standard + defense computation
  - Additional overhead: 10-20% depending on defense

### Memory
- Typical VRAM usage: 8-16GB (depends on model)
- RAM usage for metrics: ~500MB-1GB for full evaluation

## Error Handling

### Retry Logic
- Built into `query_model()` with exponential backoff
- Rate limiting via `AsyncLimiter`
- Timeout handling for long queries

### Graceful Degradation
- Failed queries skip that example but continue
- Missing defense models still run attacks
- Partial results saved if evaluation interrupted

## Integration with Existing Code

- Uses existing `TextRewards` class from `rewards/text_rewards.py`
- Compatible with OpenAI-compatible API (vLLM, local deployments)
- Integrates with existing project structure
- Outputs stored in `evaluation_results/` directory by mode

## Usage Recommendations

1. **Quick Test**: Run with 1 method, 1 model, standard mode
   ```bash
   python evaluation_method.py --method leakagent --model llama3.1-8b --n_samples 5 --top_k 2
   ```

2. **Full Evaluation**: Run comprehensive framework
   ```bash
   python run_comprehensive_evaluation.py --all
   ```

3. **Specific Analysis**: Run specific mode/defense combination
   ```bash
   python run_comprehensive_evaluation.py --method leakagent --model llama3.1-8b \
     --modes repeated_prompt --defenses promptguard
   ```

4. **Interactive**: Let the framework guide you
   ```bash
   python run_comprehensive_evaluation.py --interactive
   ```
