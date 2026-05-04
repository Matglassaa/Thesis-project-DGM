# Gemini AI Assistant Project Instructions

## Project Context
*   **Language:** Python
*   **Domain:** Geological Facies Modeling.
*   **Core Technology:** Deep Generative Models, specifically assessing Generative Adversarial Networks (GANs).
*   **Data Pipeline:** Custom 3D synthetic data generation using the `flumy` package. 
*   **Target Variables:** The primary data outputs being modeled are `age`, `facies`, and `grainsize`.

## Coding Standards & Behavior

When generating or modifying Python code in this repository, you must strictly adhere to the following rules:

### 1. Minimal Inline Comments
*   **Do not over-explain code.** Assume the user is an advanced Python developer and data scientist.
*   **No step-by-step commentary.** Do not add comments explaining standard library functions, basic loops, tensor reshaping, or standard PyTorch/TensorFlow boilerplate.
*   **Strategic comments only.** Only insert inline comments to justify specific architectural choices, explain complex domain-specific logic (e.g., handling specific `flumy` edge cases), or clarify non-obvious mathematical operations. 

### 2. Extensive Docstrings
*   **Every** function, class, and method generated must include a comprehensive docstring.
*   Use standard Google-style Python docstrings.
*   Docstrings must clearly state the purpose of the function, detail all `Args` (including expected types and tensor shapes where applicable), and specify the `Returns` (types and shapes).

### 3. Domain Awareness
*   When suggesting architectures or data preprocessing steps, prioritize spatial awareness suitable for 3D geological data.