def get_ml_problem_prompt(problem_statement, test_description, git_tempdir):
    """
    Generate ML-specific problem-solving prompt for the coding agent.
    """
    return f"""I have uploaded a Python code repository in the directory {git_tempdir}. Help solve the following machine learning problem.

<problem_description>
{problem_statement}
</problem_description>

<test_description>
{test_description}
</test_description>

Your task is to build a complete machine learning solution that addresses the <problem_description>. This includes:

1. **Data Preprocessing**:
   - Load and clean the dataset
   - Handle missing values, outliers, and data quality issues
   - Perform feature engineering and selection
   - Split data into training/validation/test sets

2. **Model Development**:
   - Select appropriate ML algorithms for the task
   - Implement model training pipeline
   - Perform hyperparameter tuning if needed
   - Handle model validation and selection

3. **Evaluation and Testing**:
   - Implement comprehensive evaluation metrics
   - Create automated testing for model performance
   - Ensure reproducibility of results
   - Generate performance reports

4. **Code Quality**:
   - Write clean, well-documented code
   - Include proper error handling
   - Create modular, reusable components
   - Add appropriate logging and monitoring

**Available Tools**:
- Use the `editor` tool to create and modify Python files
- Use the `bash` tool to run commands, install packages, and execute scripts
- You can create new tools in the `tools/` directory if needed for ML-specific tasks

**Requirements**:
- The solution should be complete and runnable
- All dependencies should be properly managed
- The evaluation should be automated and reproducible
- The final performance metric should be clearly reported

Make changes to the files in the {git_tempdir} directory to address the <problem_description> and implement ML solution. I have already taken care of the required dependencies.
"""

def get_ml_regression_test_prompt(problem_statement, test_description, git_tempdir):
    """
    Generate ML-specific regression testing prompt.
    """
    return f"""I have uploaded a Python code repository in the directory {git_tempdir}.

<problem_description>
{problem_statement}
</problem_description>

<test_description>
{test_description}
</test_description>

Your task is to identify and implement comprehensive evaluation tests for the machine learning solution in the {git_tempdir} directory that should pass before and after addressing the <problem_description>. 
This includes:

1. **Data Validation Tests**:
   - Test data loading and preprocessing
   - Validate data quality and format
   - Test feature engineering functions

2. **Model Tests**:
   - Test reproducibility of results
   - Validate model predictions
   - Test model serialization/deserialization

3. **Integration Tests**:
   - Test end-to-end model training and evaluation pipeline

At the end, please provide a summary that includes:
- Where the tests are located
- What each test validates
- How to execute the tests
- Expected test outcomes
"""

def get_ml_evaluation_prompt(problem_statement, test_description, code_diff, regression_tests_summary, git_tempdir):
    """
    Generate ML-specific evaluation prompt.
    """
    return f"""I have uploaded a Python code repository in the directory {git_tempdir}. There is an attempt to address the ML problem. Please review the changes and run the evaluation.

<problem_description>
{problem_statement}
</problem_description>

<attempted_solution>
{code_diff}
</attempted_solution>

<test_description>
{test_description}
</test_description>

<regression_tests_summary>
{regression_tests_summary}
</regression_tests_summary>

Your task is to run evaluation tests for the machine learning solution in the {git_tempdir} directory to ensure that that the changes made to the code address the <problem_description>. This includes:

1. **Data Processing Works Correctly**:
   - Data loading and preprocessing functions work
   - Feature engineering is implemented properly

2. **Model Training is Successful**:
   - Model training pipeline runs without errors
   - Hyperparameter tuning works if implemented
   - Model saves and loads correctly

3. **Evaluation is Comprehensive**:
   - All specified metrics are calculated
   - Performance meets or exceeds requirements
   - Results are reproducible

Run the evaluation and provide a detailed report of the results, including the final performance metric that will be used for scoring.
"""