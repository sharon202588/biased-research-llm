# Medical Dialogue Simulation Test System

## Project Overview
The **Medical Dialogue Simulation Test System** is a research tool designed to study the impact of **patient preferences** and **physician bias mitigation strategies** on diagnostic decision-making. The system simulates medical dialogues between patients and physicians, evaluates physician responses, and analyzes the influence of patient preferences on diagnostic accuracy.

This system supports multiple patient types and physician roles, allowing for flexible testing scenarios. It leverages AI models such as **OpenAI's ChatGPT** and **Google's Gemini** to generate realistic dialogue and responses.

---

## Features

### 1. Multi-Role Support
- **Patient Types**:
  - **Normal Patient**: Provides objective descriptions of symptoms and conditions.
  - **Confirmation Bias Patient**: Holds strong preconceived beliefs, selectively accepts information, and resists suggestions that contradict expectations.
- **Physician Roles**:
  - **Basic Physician**: Provides standard medical consultation.
  - **Bias-Aware Physician**: Focuses on identifying and mitigating patient biases.
  - **All-Inclusive Physician**: Uses a comprehensive diagnostic framework to gather additional clinical information.
  - **Step-by-Step Physician**: Follows a systematic diagnostic process to reduce cognitive bias.
  - **Few-Shot Example Physician**: Uses few-shot learning techniques to improve diagnostic accuracy.

### 2. Bias Analysis
- Automatically analyzes misleading options in test cases.
- Identifies the influence of patient preferences on physician decisions.
- Evaluates physician decision-making biases and their impact on diagnostic accuracy.

### 3. Batch Testing
- Supports single, sequential, and multi-case testing modes.
- Displays real-time results and saves them for further analysis.
- Generates detailed test reports, including error categorization and statistical summaries.

### 4. Dual AI Model Support
- Supports **OpenAI ChatGPT** and **Google Gemini** models.
- Allows users to select the AI model at the start of the test.

### 5. Static Test Mode
- Provides a non-simulated static test mode for patient scenarios.
- Directly evaluates medical questions without dialogue simulation.

---

## Workflow Overview

### 1. Initial Setup
- Select the test mode:
  - **Normal Mode**: Simulates dialogues with normal patients.
  - **Confirmation Bias Mode**: Simulates dialogues with confirmation bias patients.
  - **Static Mode**: Directly evaluates static test cases without dialogue simulation.
- Choose the physician role (e.g., Basic, Bias-Aware, All-Inclusive, Step-by-Step).
- Select the AI model (e.g., ChatGPT or Gemini).
- Choose the test cases (single, sequential, or multiple).

### 2. Case Processing
- Processes cases based on the selected test mode:
  - **Static Mode**: Directly evaluates questions and answers.
  - **Simulation Mode**: Conducts multi-turn dialogues between the patient and physician, concluding with a final diagnostic question.
- Evaluates physician responses and records results.

### 3. Result Processing
- Categorizes results by **USMLE Step** (if applicable).
- Saves detailed test data for each case.
- Outputs statistical analysis and generates detailed reports.

---

## Installation and Setup

### 1. Prerequisites
- Python 3.8 or higher
- API keys for **OpenAI ChatGPT** and/or **Google Gemini**

### 2. Install Dependencies
Install the required Python libraries using the following command:
```bash
pip install -r requirements.txt
```

### 3. Configure API Keys
- Open the `config.py` file.
- Set the `PATIENT_API_KEY` and `DOCTOR_CHATGPT_API_KEY` with your valid API keys.
- Example:
  ```python
  PATIENT_API_KEY = "your_openai_api_key"
  DOCTOR_CHATGPT_API_KEY = "your_openai_api_key"
  ```

### 4. Run the Program
Execute the main program file:
```bash
python "physician_patient_interactions-based_evaluation.py"
```

---

## File Structure

```
.
├── physician_patient_interactions-based_evaluation.py  # Main program file
├── role_prompts.py                        # Role definitions for patients and physicians
├── config.py                              # Configuration file for API keys and settings
├── requirements.txt                       # Python dependencies
└── README.md                              # Project documentation
```

---

## Example Output

### Evaluation Results
```plaintext
Evaluation Result: (Based on Valid Cases)
  Correct Diagnosis Count: 10
  Patient Preference Influenced Error Count: 2
  Incorrect Answer from Other Reasons: 3

Result Analysis:
  - Correct Diagnosis Rate: 71.4%
  - Preference Influenced Error Rate: 14.3%
  - Other Error Rate: 14.3%
```

---

## Notes

1. **API Key Security**:
   - Do not share your API keys publicly.
   - Ensure your API keys have sufficient usage limits for the test cases.

2. **Research Use Only**:
   - This system is designed for research purposes and should not be used for actual medical diagnosis.

3. **Debugging**:
   - Enable debugging mode in `config.py` by setting `DEBUG = True` to view detailed logs during execution.

---

## Future Improvements

1. Add support for additional patient and physician roles.
2. Implement a web-based interface for easier interaction.
3. Enhance evaluation metrics for more detailed analysis.
4. Expand the static test mode with more diverse test cases.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.