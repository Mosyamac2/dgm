

def get_test_command(eval_script):
    test_hint = ''
    # test_command is the 2nd last line in eval_script
    lines = eval_script.strip().split('\n')
    test_command = lines[-2].strip()
    # Remove trailing arguments specifying filepaths
    parts = test_command.split()
    if '.' in parts[-1] and not parts[-1].endswith('.py'):
        # Get the test hint
        test_hint = 'If the target test file path is tests/some_folder/some_file.py, then <specific test files> should be `some_folder.some_file`.'
    while parts and '.' in parts[-1]:
        parts.pop()
    # Reconstruct the command
    test_command = ' '.join(parts)
    return f'cd /testbed/ && {test_command} <specific test files>', test_hint

def get_test_description(eval_script='', swerepo=False, polyglot=False, ml_task=False):
    assert not (swerepo and polyglot and ml_task), "swerepo and polyglot and ml_task cannot simultaniously be True"
    if swerepo:  # SWE repo
        swe_prompt = '''The tests in the repository can be run with the bash command `{test_command}`. If no specific test files are provided, all tests will be run. The given command-line options must be used EXACTLY as specified. Do not use any other command-line options. {test_hint}'''
        test_command, test_hint = get_test_command(eval_script)
        description = swe_prompt.format(test_command=test_command, test_hint=test_hint)
    elif polyglot:
        description = f"In the repository folder, the tests can be run with the following bash command(s):\n\n```{eval_script}```\n"
    elif ml_task:
        description = '''For machine learning tasks, evaluation is based on model performance metrics. 

        The ML model should be evaluated using the following approach:
        1. **Data Loading**: Load the dataset specified in the problem statement
        2. **Model Training**: Train the model using appropriate algorithms
        3. **Model Evaluation**: Evaluate using the specified metric (accuracy, precision, recall, etc.)
        4. **Results Reporting**: Report the final performance score

        Evaluation Command: The model should be evaluated by running the training and evaluation script. 
        The evaluation script should output the final performance metric that will be used for scoring.

        Expected Output Format:
        Model Performance: <metric_name> = <score>
        Training Time: <time>
        Test Accuracy: <accuracy>
        The evaluation should be automated and reproducible.'''

    else:  # dgm repo
        description = 'The tests in the repository can be run with the bash command `cd /dgm/ && pytest -rA <specific test files>`. If no specific test files are provided, all tests will be run. The given command-line options must be used EXACTLY as specified. Do not use any other command-line options. ONLY test tools and utils. NEVER try to test or run agentic_system.forward().'

    return description.strip()
