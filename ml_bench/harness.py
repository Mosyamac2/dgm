import json
import subprocess
import tempfile
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits, load_diabetes
import pickle
import sys
import traceback
import warnings
warnings.filterwarnings('ignore')

def load_ml_dataset(dataset_info):
    """Load ML dataset based on dataset information"""
    dataset_id = dataset_info['dataset_id']
    
    # Load dataset based on sklearn dataset name
    if 'iris' in dataset_id.lower():
        data = load_iris()
    elif 'wine' in dataset_id.lower():
        data = load_wine()
    elif 'breast_cancer' in dataset_id.lower():
        data = load_breast_cancer()
    elif 'digits' in dataset_id.lower():
        data = load_digits()
    elif 'diabetes' in dataset_id.lower():
        data = load_diabetes()
    else:
        # Default to iris for unknown datasets
        data = load_iris()
    
    # Convert to DataFrame
    X = pd.DataFrame(data.data, columns=data.feature_names if hasattr(data, 'feature_names') else [f'feature_{i}' for i in range(data.data.shape[1])])
    
    if hasattr(data, 'target_names'):
        y = pd.Series(data.target, name=dataset_info.get('target', 'target'))
    else:
        y = pd.Series(data.target, name=dataset_info.get('target', 'target'))
    
    return X, y, dataset_info

def evaluate_ml_model(dataset_info, model_code, temp_dir):
    """Evaluate ML model code on dataset"""
    try:
        # Setup environment
        metadata_path, train_path, test_path = create_ml_environment(dataset_info, temp_dir)
        
        # Create template ML code that loads data and expects model implementation
        template_code = f"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import json
import warnings
warnings.filterwarnings('ignore')

# Load metadata
with open('dataset_metadata.json', 'r') as f:
    metadata = json.load(f)

# Load data
train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data.csv')

target_col = metadata['target']
feature_cols = metadata['features']

X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_test = test_df[feature_cols] 
y_test = test_df[target_col]

task_type = metadata['task_type']

print("Dataset loaded successfully")
print(f"Training set shape: {{X_train.shape}}")
print(f"Test set shape: {{X_test.shape}}")
print(f"Task type: {{task_type}}")

# User's model code will be inserted here
{model_code}

# Evaluate the model
try:
    if hasattr(locals().get('model'), 'predict'):
        y_pred = model.predict(X_test)
    else:
        print("ERROR: No trained model found or model doesn't have predict method")
        exit(1)
        
    # Calculate metrics based on task type
    if task_type == 'classification':
        accuracy = accuracy_score(y_test, y_pred)
        try:
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
        except:
            f1 = precision = recall = 0.0
        
        print(f"Accuracy: {{accuracy:.4f}}")
        print(f"F1 Score: {{f1:.4f}}")
        
        # Save results
        results = {{
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'primary_metric': accuracy
        }}
        
    elif task_type == 'regression':
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"MSE: {{mse:.4f}}")
        print(f"RMSE: {{rmse:.4f}}")
        print(f"R² Score: {{r2:.4f}}")
        
        # Save results
        results = {{
            'mse': mse,
            'rmse': rmse,
            'r2_score': r2,
            'primary_metric': r2
        }}
    
    # Save results to file
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print("Model evaluation completed successfully")
    
except Exception as e:
    print(f"ERROR in evaluation: {{str(e)}}")
    import traceback
    traceback.print_exc()
    exit(1)
"""
        
        # Execute the combined code
        success, stdout, stderr = execute_ml_code(template_code, temp_dir)
        
        # Parse results
        results_file = os.path.join(temp_dir, 'results.json')
        if success and os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            return True, results, stdout, stderr
        else:
            return False, None, stdout, stderr
            
    except Exception as e:
        return False, None, "", str(e)

def harness(test_task_list, model_name_or_path, model_patch_paths, 
           num_evals, pred_dname, **kwargs):
    """Main harness function for ML evaluation"""
    
    results = []
    
    for task_entry in test_task_list:
        # Load dataset info from task entry
        if isinstance(task_entry, str):
            # If it's a string, treat it as a dataset ID and create minimal info
            dataset_info = {
                'dataset_id': task_entry,
                'task_type': 'classification',
                'metric': 'accuracy',
                'baseline_accuracy': 0.5
            }
        else:
            dataset_info = task_entry
        
        print(f"Evaluating on dataset: {dataset_info['dataset_id']}")
        
        # Create temporary directory for this evaluation
        with tempfile.TemporaryDirectory() as temp_dir:
            
            # For multiple evaluations
            eval_results = []
            
            for eval_num in range(num_evals):
                print(f"Evaluation {eval_num + 1}/{num_evals}")
                
                # Read model code from patches or use default
                model_code = ""
                if model_patch_paths:
                    for patch_path in model_patch_paths:
                        if os.path.exists(patch_path):
                            # Extract Python code from patch file
                            with open(patch_path, 'r') as f:
                                patch_content = f.read()
                            
                            # Simple extraction of Python code from patch
                            lines = patch_content.split('\n')
                            code_lines = []
                            
                            for line in lines:
                                if line.startswith('+') and not line.startswith('+++'):
                                    code_line = line[1:].strip()
                                    if code_line and not code_line.startswith('#'):
                                        code_lines.append(code_line)
                            
                            model_code = '\n'.join(code_lines)
                
                if not model_code.strip():
                    # Default simple model code for testing
                    model_code = """
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

if task_type == 'classification':
    model = RandomForestClassifier(random_state=42)
else:
    model = RandomForestRegressor(random_state=42)

model.fit(X_train, y_train)
"""
                
                # Evaluate the model
                success, eval_result, stdout, stderr = evaluate_ml_model(dataset_info, model_code, temp_dir)
                
                eval_results.append({
                    'success': success,
                    'result': eval_result,
                    'stdout': stdout,
                    'stderr': stderr
                })
                
                if not success:
                    print(f"Evaluation failed: {stderr}")
            
            # Aggregate results
            successful_evals = [r for r in eval_results if r['success']]
            
            if successful_evals:
                # Average the metrics across successful evaluations
                primary_metrics = [r['result']['primary_metric'] for r in successful_evals]
                avg_metric = np.mean(primary_metrics)
                
                task_result = {
                    'dataset_id': dataset_info['dataset_id'],
                    'model_name': model_name_or_path,
                    'num_evals': len(successful_evals),
                    'success_rate': len(successful_evals) / num_evals,
                    'average_metric': avg_metric,
                    'individual_results': eval_results,
                    'task_type': dataset_info['task_type']
                }
            else:
                task_result = {
                    'dataset_id': dataset_info['dataset_id'],
                    'model_name': model_name_or_path,
                    'num_evals': 0,
                    'success_rate': 0.0,
                    'average_metric': 0.0,
                    'individual_results': eval_results,
                    'task_type': dataset_info.get('task_type', 'classification')
                }
            
            results.append(task_result)
    
    # Save overall results
    os.makedirs(pred_dname, exist_ok=True)
    results_file = os.path.join(pred_dname, f"{model_name_or_path}_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"ML evaluation completed. Results saved to {results_file}")
    
    # Return list of result directories (mimicking SWE-bench structure)
    return [pred_dname]
