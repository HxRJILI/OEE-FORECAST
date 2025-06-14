Contributing Guide
==================

Thank you for your interest in contributing to the OEE Forecasting and Analytics project! This guide will help you understand how to contribute effectively to make this project better for the manufacturing community.

🤝 **Ways to Contribute**
=========================

We welcome various types of contributions from the community:

**Code Contributions**
---------------------
- 🐛 **Bug Fixes**: Fix issues and improve stability
- ✨ **New Features**: Add new functionality and capabilities
- 🔧 **Improvements**: Enhance existing features and performance
- 📚 **Documentation**: Improve and expand documentation
- 🧪 **Tests**: Add or improve test coverage

**Non-Code Contributions**
-------------------------
- 📝 **Documentation**: Write tutorials, guides, and examples
- 🎨 **Design**: Improve UI/UX and create visual assets
- 🔍 **Testing**: Test new features and report bugs
- 💬 **Community**: Help others in discussions and forums
- 🌐 **Translation**: Translate documentation and interface
- 📊 **Data**: Contribute sample datasets or use cases

**Research and Analysis**
------------------------
- 🔬 **Model Research**: Develop new forecasting algorithms
- 📈 **Performance Analysis**: Benchmark and optimize existing models
- 🏭 **Industry Insights**: Share manufacturing domain expertise
- 📋 **Use Cases**: Document real-world implementation scenarios

🚀 **Getting Started**
======================

**Prerequisites**
----------------

Before contributing, ensure you have:

- **Python 3.8+** installed
- **Git** for version control
- **GitHub account** for collaboration
- **Basic knowledge** of:
  - Python programming
  - Manufacturing concepts (OEE, production metrics)
  - Machine learning (for model contributions)
  - Web development (for UI contributions)

**Development Setup**
--------------------

1. **Fork and Clone the Repository:**

   .. code-block:: bash

      # Fork the repository on GitHub first, then clone your fork
      git clone https://github.com/YOUR_USERNAME/OEE-FORECAST.git
      cd OEE-FORECAST

      # Add upstream remote
      git remote add upstream https://github.com/HxRJILI/OEE-FORECAST.git

2. **Set Up Development Environment:**

   .. code-block:: bash

      # Create virtual environment
      python -m venv oee_dev_env
      source oee_dev_env/bin/activate  # Linux/Mac
      # or
      oee_dev_env\Scripts\activate     # Windows

      # Install dependencies
      pip install -r requirements.txt
      pip install -r requirements_rag.txt
      pip install -r requirements_dev.txt  # Development dependencies

3. **Install Development Tools:**

   .. code-block:: bash

      # Install pre-commit hooks
      pre-commit install

      # Install testing tools
      pip install pytest pytest-cov black flake8 mypy

4. **Verify Setup:**

   .. code-block:: bash

      # Run tests to ensure everything works
      pytest tests/

      # Run the application
      streamlit run app.py

📋 **Development Workflow**
==========================

**Branch Strategy**
------------------

We use a feature branch workflow:

.. code-block::

   Branch Structure:
   
   main
   ├── develop              # Development branch
   ├── feature/new-model    # Feature branches
   ├── bugfix/fix-issue-123 # Bug fix branches
   ├── hotfix/critical-fix  # Critical fixes
   └── release/v2.2.0       # Release branches

**Creating a Feature Branch**
----------------------------

.. code-block:: bash

   # Start from develop branch
   git checkout develop
   git pull upstream develop

   # Create and switch to feature branch
   git checkout -b feature/your-feature-name

   # Make your changes and commit
   git add .
   git commit -m "Add: Brief description of your changes"

   # Push to your fork
   git push origin feature/your-feature-name

**Commit Message Guidelines**
----------------------------

Use clear, descriptive commit messages following this format:

.. code-block::

   Type: Brief description (50 characters or less)

   Detailed explanation if needed (wrap at 72 characters)

   Types:
   - Add: New features or functionality
   - Fix: Bug fixes
   - Update: Improvements to existing features
   - Remove: Removing code or features
   - Docs: Documentation changes
   - Style: Code style changes (formatting, etc.)
   - Refactor: Code refactoring without feature changes
   - Test: Adding or updating tests

**Examples:**

.. code-block::

   Add: Multi-Kernel CNN model for improved forecasting accuracy

   Implement new deep learning architecture with parallel convolutional
   branches for better pattern recognition. Achieves 15% improvement
   in MAPE scores across all production lines.

   Fix: Resolve memory leak in data processing pipeline

   Update: Enhance Streamlit UI responsiveness for mobile devices

🧪 **Testing Guidelines**
=========================

**Testing Philosophy**
---------------------

We maintain high code quality through comprehensive testing:

- **Unit Tests**: Test individual functions and methods
- **Integration Tests**: Test component interactions
- **Performance Tests**: Ensure acceptable performance
- **User Acceptance Tests**: Validate user workflows

**Writing Tests**
----------------

**Unit Test Example:**

.. code-block:: python

   # tests/test_oee_calculator.py
   import pytest
   import pandas as pd
   from datetime import date
   from src.oee_calculator import OEECalculator

   class TestOEECalculator:
       def setup_method(self):
           """Setup test fixtures"""
           self.calculator = OEECalculator()
           self.sample_data = pd.DataFrame({
               'START_DATETIME': ['2024-01-01 08:00:00'],
               'PRODUCTION_LINE': ['LINE-01'],
               'STATUS_NAME': ['Production']
           })

       def test_calculate_availability_basic(self):
           """Test basic availability calculation"""
           availability = self.calculator.calculate_availability(
               self.sample_data, 'LINE-01', date(2024, 1, 1)
           )
           assert 0 <= availability <= 1
           assert isinstance(availability, float)

       def test_calculate_availability_invalid_line(self):
           """Test availability calculation with invalid line"""
           availability = self.calculator.calculate_availability(
               self.sample_data, 'INVALID-LINE', date(2024, 1, 1)
           )
           assert availability == 0.0

       @pytest.mark.parametrize("line,expected_range", [
           ('LINE-01', (0.6, 0.9)),
           ('LINE-03', (0.7, 0.95)),
           ('LINE-06', (0.8, 0.98))
       ])
       def test_availability_ranges_by_line(self, line, expected_range):
           """Test availability ranges for different production lines"""
           # Test with realistic data
           availability = self.calculator.calculate_availability(
               self.sample_data, line, date(2024, 1, 1)
           )
           assert expected_range[0] <= availability <= expected_range[1]

**Integration Test Example:**

.. code-block:: python

   # tests/test_forecasting_integration.py
   import pytest
   import numpy as np
   from src.forecasting import OEEForecaster
   from src.data_processing import load_and_process_data

   class TestForecastingIntegration:
       def test_end_to_end_forecasting(self):
           """Test complete forecasting pipeline"""
           
           # Load and process data
           line_status, production_data = load_and_process_data(
               'tests/fixtures/sample_line_status.csv',
               'tests/fixtures/sample_production.csv'
           )
           
           # Initialize forecaster
           forecaster = OEEForecaster(model_type='multi_kernel_cnn')
           
           # Train model
           training_results = forecaster.fit(line_status, production_line='LINE-01')
           assert training_results['performance_metrics']['mae'] < 0.15
           
           # Generate predictions
           predictions = forecaster.predict(steps=7)
           assert len(predictions['forecasts']) == 7
           assert all(0 <= pred <= 1 for pred in predictions['forecasts'])

**Running Tests**
----------------

.. code-block:: bash

   # Run all tests
   pytest

   # Run with coverage report
   pytest --cov=src/ --cov-report=html

   # Run specific test file
   pytest tests/test_oee_calculator.py

   # Run tests with specific marker
   pytest -m "not slow"

   # Run tests in parallel
   pytest -n auto

📝 **Code Style and Standards**
==============================

**Python Style Guidelines**
---------------------------

We follow PEP 8 with some project-specific conventions:

.. code-block:: python

   # Good Examples

   def calculate_oee_metrics(line_data, production_data, date_range):
       """
       Calculate OEE metrics for specified date range.
       
       Args:
           line_data (pd.DataFrame): Production line status data
           production_data (pd.DataFrame): Production output data
           date_range (tuple): Start and end dates
       
       Returns:
           dict: OEE metrics including availability, performance, quality
       
       Raises:
           ValueError: If date_range is invalid
           DataProcessingError: If data is corrupted
       """
       
       # Input validation
       if not isinstance(line_data, pd.DataFrame):
           raise TypeError("line_data must be a pandas DataFrame")
       
       # Clear variable names
       start_date, end_date = date_range
       filtered_data = line_data[
           (line_data['date'] >= start_date) & 
           (line_data['date'] <= end_date)
       ]
       
       # Use descriptive constants
       SECONDS_PER_HOUR = 3600
       PLANNED_PRODUCTION_HOURS = 16
       
       # Calculate metrics
       availability = calculate_availability(filtered_data)
       performance = calculate_performance(filtered_data, production_data)
       quality = calculate_quality(production_data)
       
       return {
           'availability': availability,
           'performance': performance,
           'quality': quality,
           'oee': availability * performance * quality
       }

**Code Formatting**
------------------

We use automated code formatting tools:

.. code-block:: bash

   # Format code with black
   black src/ tests/

   # Check formatting
   black --check src/ tests/

   # Sort imports
   isort src/ tests/

   # Lint code
   flake8 src/ tests/

   # Type checking
   mypy src/

**Documentation Standards**
--------------------------

**Docstring Format:**

.. code-block:: python

   def complex_function(param1, param2, optional_param=None):
       """
       Brief description of what the function does.
       
       Longer description if needed, explaining the purpose,
       behavior, and any important details.
       
       Args:
           param1 (type): Description of parameter
           param2 (type): Description of parameter
           optional_param (type, optional): Description. Defaults to None.
       
       Returns:
           type: Description of return value
       
       Raises:
           ExceptionType: Description of when this exception is raised
       
       Example:
           >>> result = complex_function('input1', 42)
           >>> print(result)
           Expected output
       
       Note:
           Any important notes or warnings
       """

🐛 **Reporting Issues**
======================

**Before Reporting**
-------------------

1. **Search Existing Issues**: Check if the issue is already reported
2. **Check Documentation**: Ensure it's not a known limitation
3. **Test with Latest Version**: Verify the issue exists in the current version
4. **Reproduce Consistently**: Ensure you can reproduce the issue

**Issue Report Template**
------------------------

When reporting bugs, please use this template:

.. code-block::

   **Bug Description**
   A clear description of what the bug is.

   **Steps to Reproduce**
   1. Go to '...'
   2. Click on '...'
   3. Enter data '...'
   4. See error

   **Expected Behavior**
   What you expected to happen.

   **Actual Behavior**
   What actually happened.

   **Environment**
   - OS: [e.g., Ubuntu 20.04, Windows 10, macOS 12]
   - Python Version: [e.g., 3.9.7]
   - Project Version: [e.g., 2.1.0]
   - Browser (if applicable): [e.g., Chrome 96, Firefox 95]

   **Additional Context**
   - Error messages or logs
   - Screenshots if applicable
   - Sample data if relevant (anonymized)

**Feature Request Template**
---------------------------

.. code-block::

   **Feature Summary**
   Brief description of the proposed feature.

   **Problem Statement**
   What problem does this feature solve?

   **Proposed Solution**
   Detailed description of the proposed feature.

   **Alternatives Considered**
   Other solutions you've considered.

   **Use Cases**
   Specific scenarios where this feature would be valuable.

   **Implementation Notes**
   Any technical considerations or constraints.

🔄 **Pull Request Process**
==========================

**Before Submitting**
--------------------

1. **Update Your Branch:**

   .. code-block:: bash

      git checkout develop
      git pull upstream develop
      git checkout your-feature-branch
      git merge develop

2. **Run All Tests:**

   .. code-block:: bash

      pytest
      flake8 src/ tests/
      black --check src/ tests/
      mypy src/

3. **Update Documentation:**
   - Add or update docstrings
   - Update relevant documentation files
   - Add examples if applicable

**Pull Request Template**
------------------------

.. code-block::

   ## Description
   Brief description of changes and motivation.

   ## Type of Change
   - [ ] Bug fix (non-breaking change that fixes an issue)
   - [ ] New feature (non-breaking change that adds functionality)
   - [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
   - [ ] Documentation update
   - [ ] Performance improvement
   - [ ] Code refactoring

   ## Testing
   - [ ] Added tests for new functionality
   - [ ] All existing tests pass
   - [ ] Manual testing completed

   ## Documentation
   - [ ] Code is documented with docstrings
   - [ ] Documentation updated (if applicable)
   - [ ] Examples added (if applicable)

   ## Performance Impact
   Describe any performance implications.

   ## Breaking Changes
   List any breaking changes and migration steps.

   ## Screenshots (if applicable)
   Add screenshots for UI changes.

**Review Process**
-----------------

1. **Automated Checks**: All CI checks must pass
2. **Code Review**: At least one maintainer review required
3. **Testing**: Reviewers may test the changes locally
4. **Documentation Review**: Ensure documentation is clear and complete
5. **Approval**: Maintainer approval required for merge

👥 **Community Guidelines**
==========================

**Code of Conduct**
------------------

We are committed to providing a welcoming and inclusive environment:

- **Be Respectful**: Treat everyone with respect and professionalism
- **Be Collaborative**: Work together to improve the project
- **Be Inclusive**: Welcome newcomers and diverse perspectives
- **Be Constructive**: Provide helpful feedback and suggestions
- **Be Patient**: Remember that everyone has different experience levels

**Communication Channels**
-------------------------

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community discussions
- **Pull Requests**: Code review and collaboration
- **Email**: Direct contact for sensitive issues

**Recognition**
--------------

We value all contributions and recognize contributors through:

- **Contributors List**: Listed in README and documentation
- **Changelog**: Credited in release notes
- **Community Spotlight**: Featured in project updates
- **Commit Attribution**: Proper attribution in git history

🎯 **Specific Contribution Areas**
==================================

**Manufacturing Domain Expertise**
---------------------------------

We especially welcome contributions from manufacturing professionals:

- **OEE Best Practices**: Share industry knowledge and standards
- **Real-World Use Cases**: Document actual implementation scenarios
- **Data Patterns**: Contribute insights about production data characteristics
- **Validation**: Help validate model accuracy against real-world results

**Machine Learning and Data Science**
------------------------------------

Areas where ML expertise is valuable:

- **New Models**: Develop advanced forecasting algorithms
- **Optimization**: Improve model performance and efficiency
- **Evaluation**: Enhance model validation and testing procedures
- **Research**: Investigate cutting-edge approaches for time series forecasting

**Software Engineering**
-----------------------

Technical improvements needed:

- **Performance**: Optimize code for speed and memory usage
- **Architecture**: Improve system design and modularity
- **Testing**: Expand test coverage and automation
- **Infrastructure**: Enhance deployment and monitoring capabilities

📚 **Learning Resources**
========================

**Project-Specific Resources**
-----------------------------

- **Documentation**: Complete project documentation
- **Tutorials**: Step-by-step guides for common tasks
- **API Reference**: Detailed API documentation
- **Examples**: Sample implementations and use cases

**External Learning**
--------------------

**Manufacturing and OEE:**
- MESA International (Manufacturing Enterprise Solutions Association)
- SEMI Standards for OEE calculation
- Lean Manufacturing principles and practices

**Machine Learning:**
- TensorFlow and Keras documentation
- Time series forecasting tutorials
- Deep learning for manufacturing applications

**Software Development:**
- Python best practices and PEP standards
- Streamlit documentation and tutorials
- Git workflow and collaboration techniques

🏆 **Contributor Recognition**
=============================

**Contribution Levels**
----------------------

.. list-table:: Contributor Levels
   :header-rows: 1
   :widths: 20 30 50

   * - Level
     - Criteria
     - Recognition
   * - **Contributor**
     - First merged PR
     - Listed in contributors, thank you message
   * - **Regular Contributor**
     - 5+ merged PRs
     - Featured in release notes, priority review
   * - **Core Contributor**
     - 20+ PRs, sustained involvement
     - Commit access, release planning input
   * - **Maintainer**
     - Long-term commitment, leadership
     - Full repository access, decision-making role

**Special Recognitions**
-----------------------

- **Bug Hunter**: Exceptional bug finding and reporting
- **Documentation Champion**: Outstanding documentation contributions
- **Performance Hero**: Significant performance improvements
- **Innovation Award**: Novel features or approaches
- **Community Builder**: Exceptional community support and mentoring

Thank you for contributing to the OEE Forecasting and Analytics project! Together, we're building tools that help manufacturers optimize their operations and improve efficiency worldwide. 

For questions about contributing, please reach out through our communication channels or check our FAQ in the GitHub Discussions.