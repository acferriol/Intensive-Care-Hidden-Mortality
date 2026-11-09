# Summary of work completed in Intensive-Care-Hidden-Mortality

Prepared on November 9, 2026.

This document brings together the work recorded in the repository files and Git history. Recorded development began on January 11, 2024, and the latest available commit is `2cf8140`, dated December 18, 2025. The document's filename refers to the year this summary was prepared; it does not imply that new implementations were recorded during 2026.

The review draws on source code, notebook cells and saved outputs, result and configuration files, versioned artifacts, and commit messages. No models were retrained and no AI service requests were made to prepare it. Reported metrics are preserved historical results, not new measurements.

## Continuation of the work and integration into SADUCI

This project's predictive capability is already integrated into the public [coslatte/saduci](https://github.com/coslatte/saduci) repository. Its prediction service uses a byte-identical copy of `new_workflow.joblib`, with HTTP contracts, persistence, and narrative generation. This identifies a concrete destination for integrated model execution without implying that every notebook or explainer has been transferred.

The [integration and transfer report](IntegracionSADUCI2026.md) describes which elements remain as research in this repository, which have been incorporated or adapted in SADUCI, and which differences need updating. It also distinguishes that functional transfer from the contribution of these reports to the original repository through a fork. The documentation preserves the experiments' origin and links to their continuation in the integrated system.

## 1. Project purpose and scope

The project studies mortality prediction for intensive care unit patients using machine learning. Its development combines clinical data experiments, classifier comparisons, class imbalance handling, and explanations of individual predictions. A web interface was subsequently added to enter patient characteristics and obtain an estimated probability of non-survival at ICU discharge.

The term “hidden mortality” appears in the context and language-model instructions. In the reviewed training workflows, the target variable is `Fallece`. Those workflows do not define a separate label independently identifying hidden mortality. The documented computational output is therefore a mortality prediction interpreted within that research context.

The repository preserves several generations of experiments. Some notebooks contain early comparisons, others explore explainability, and later ones construct reusable preprocessing and classification workflows. The application brings part of this work into an interactive prototype.

## 2. Development history

### January 2024: data preparation and initial models

The first commit, `bdba1c0`, introduced classification and regression notebooks. Work over the following days covered data inspection, missing-value checks, and construction of the feature set supplied to the algorithms.

The history of `My_regression.ipynb`, later reorganized, records separating `Fallece` from the predictors, selecting individual observations, and exploring class frequencies. Bar charts were added to display that distribution.

Between January 15 and 17, SMOTE, training/test splitting, logistic regression, predictions, metrics, and cross-validation were introduced. This established the project's initial experimental workflow.

### March 2024: partitions and systematic comparison

The notebooks were reorganized and training, validation, and test sets were introduced. Commit `632585f`, dated March 19, documents an 80:10:10 split.

Commits on March 29 and 30 record several versions prepared for a forum. One, `cff8265`, describes removing dummy variables in that version and using F1 for hyperparameter search. These decisions belong to a particular stage: later workflows introduce explicit categorical encoding again.

### April 2025: explainability, interface, and a new workflow

April 2025 expanded comparisons without oversampling and consolidated work visible in the application. Several data files and Excel exports were removed from tracking in `9fa32a3`, dated April 4.

A development container was added on April 9. On April 15, diagnosis selection was improved using descriptions, and the interface's explanation flow was changed. Between April 18 and 21, a new workflow notebook was introduced, `new_workflow.joblib` was saved, and the application was updated to use it.

### May 2025: inference and data augmentation

On May 15, `inference.ipynb` was added and comparisons of data augmentation techniques were expanded in `test-workflow.ipynb`. Work included probability inspection, error review, and evaluation with different oversampling variants.

### June 2025: logistic regression and narrative explanations

On June 18, a dedicated logistic regression workflow was added, and the MLP and logistic regression notebooks were separated. `new_workflow_lr.joblib` was also saved.

On June 21, the interface language changed and DataFrame presentation was adjusted. On June 28 and 29, language-model experiments were added in `llms/model-explain-text.ipynb`, narrative generation was incorporated into the application, and API-key configuration was corrected. Several revisions during the same period adjusted dependencies and resolved environment incompatibilities.

### December 2025: Gemini integration

Commit `2cf8140`, dated December 18, updated `web_app.py` and `requirements.txt` for Gemini integration. The reviewed version requests narratives from `gemini-2.5-flash-lite` through `google.generativeai`. This describes the preserved code, not a current service availability check.

## 3. Clinical data and preparation

The notebooks primarily load `Datos_regresion.csv`. Field tests use `prueba_de_campo.xls` and, in later workflows, `pc.csv`. Git configuration excludes CSV, XLS, and XLSX files, so these datasets are not part of the versioned data reviewed.

The workflow used by the application accepts these features:

| Variable | Meaning in the project |
| --- | --- |
| `Edad` | Patient age. |
| `Diag.Ing1` | First admission diagnosis. |
| `Diag.Ing2` | Second admission diagnosis. |
| `Diag.Egr2` | Second discharge diagnosis. |
| `APACHE` | Score presented as APACHE II in the interface. |
| `TiempoVAM` | Duration of mechanical ventilation. |

Early analyses also use `Diag.Egr1`. Variants were compared with all features, without the first discharge diagnosis, and without both discharge diagnoses. The current application retains `Diag.Egr2` and excludes `Diag.Egr1`.

Exploratory analysis includes descriptive statistics, data types, diagnosis frequencies, and tables crossing diagnoses with `Fallece`. `Classification.ipynb` inspects `TiempoVAM` values greater than 120 and removes row 207. This helps explain why some outputs show 207 observations and others 208.

One classification output records 176 class-0 and 31 class-1 observations before resampling. `inference.ipynb` shows 176 class-0 and 32 class-1 observations. These figures belong to different dataset states and should not be combined as though they came from one execution.

Field-test preparation removes columns including position, sex, skin, admission and discharge dates, transfer status, origin, and ventilation modes. The newer workflows also divide `TiempoVAM` by 24 when preparing `pc.csv`. This conversion must be accounted for when interpreting model units.

## 4. Class imbalance handling

The imbalance between survival and mortality motivated a substantial part of the experiments. Workflows were created without oversampling and with Random Over Sampling and SMOTE. Later stages added ADASYN and BorderlineSMOTE.

Random Over Sampling duplicates minority-class observations. SMOTE and related techniques generate synthetic observations. The notebooks compare their effects on mortality detection and errors involving survivors.

These experimental lines are preserved separately:

| Notebook | Main work |
| --- | --- |
| `Classification_RandomOverSampling.ipynb` | ROS experiments and additional preprocessing and resampling comparisons. |
| `Classification_SMOTE.ipynb` | Synthetic data generation experiments using SMOTE. |
| `Classification_without_oversampling.ipynb` | Comparison without oversampling, with hyperparameter search. |
| `Classification_without_oversampling_without_gridsearch.ipynb` | Baseline without oversampling or exhaustive search. |

Some early workflows resample the full dataset before splitting it. This can create dependencies between training and evaluation, for example when copies of an observation reach different partitions. Later workflows split training and test data first and generate augmented datasets from training data.

Some experiments also apply synthetic methods to numerically encoded diagnosis columns before categorical encoding. The clinical validity of interpolated values in those columns requires specific review. Recording these experiments does not establish that all synthetic observations represent valid diagnostic categories.

## 5. Models and hyperparameter search

The main classifier families are K nearest neighbors, logistic regression, and multilayer perceptrons.

KNN experiments vary neighbor count, uniform or distance weighting, search algorithm, leaf size, and Minkowski distance. Logistic regression experiments vary `C`, penalty, and solver. MLP experiments vary hidden-layer sizes, activations, optimizers, regularization, and maximum iterations.

Early notebooks use `GridSearchCV` with five folds and F1 as the selection criterion. `Classification.ipynb` also includes ten-fold stratified cross-validation and training and test metrics.

Metrics include recall, F1, precision, confusion matrices, balanced accuracy, Cohen's kappa, and values labeled AUC-ROC. Considering these together reveals problems that overall accuracy can conceal in an imbalanced dataset.

`configuracion_ROS.txt` and `configuracion_SMOTE.txt` preserve hyperparameters for different feature selections. For example, the first ROS block records a one-hidden-layer MLP with 50 neurons and `tanh`, while the first SMOTE block records 100 neurons and `relu`. These are historical configurations and do not necessarily describe the model currently loaded by the interface.

## 6. Historical ROS and SMOTE results

The text files contain multiple blocks, some repeated. This table reproduces only the first block labeled `Con todos los rasgos`, meaning all features, in each file. Vectors retain their source order.

| Method | Model | Recall by class | F1 by class | Value labeled AUC-ROC |
| --- | --- | --- | --- | --- |
| ROS | KNN | 1.00 / 0.85 | 0.9091 / 0.9189 | 0.9250 |
| ROS | Logistic regression | 0.80 / 0.70 | 0.7273 / 0.7568 | 0.7500 |
| ROS | MLP | 1.00 / 0.90 | 0.9375 / 0.9474 | 0.9500 |
| SMOTE | KNN | 0.8667 / 0.75 | 0.7879 / 0.8108 | 0.8083 |
| SMOTE | Logistic regression | 1.00 / 0.75 | 0.8571 / 0.8571 | 0.8750 |
| SMOTE | MLP | 0.80 / 0.90 | 0.8276 / 0.8780 | 0.8500 |

Removing diagnoses changes these results. For example, the SMOTE block without `Diag.Egr1` records MLP recall of 0.9333 and 0.90, F1 of 0.9032 and 0.9231, and a value labeled AUC-ROC of 0.9167.

These files document executions and help reconstruct comparisons. Repeated blocks and the absence of unique run identifiers prevent treating them as a normalized experiment registry or confidently linking each block to a serialized artifact.

Several cells calculate `roc_auc_score` using `y_pred`, meaning discrete labels. These values are not equivalent to evaluating probability ranking through a full ROC curve. The original label is therefore retained with this qualification; the highest value is not presented as definitive evidence of clinical superiority.

## 7. New preprocessing and training workflows

`test-workflow.ipynb`, `test-mlp-workflow.ipynb`, and `test-lr-workflow.ipynb` organize a later generation of experiments. Diagnoses are treated as categorical features, while age, APACHE, and ventilation duration are numerical.

Preprocessing uses `ColumnTransformer`, `OneHotEncoder` with `handle_unknown="infrequent_if_exist"`, and `RobustScaler`. A pipeline keeps column transformations and the classifier together.

The main split reserves 10% for testing, stratified by `Fallece`, with random seed 42. The MLP in that comparison has hidden layers of 200, 150, 100, and 50 neurons, ReLU activation, the Adam optimizer, early stopping, and up to 1000 iterations.

The notebooks generate training sets using ADASYN, SMOTE, BorderlineSMOTE, and ROS and refit the pipeline with each variant. Logistic regression uses a similar structure for comparison. Its classifier step retains the internal name `mlp` even though the object is `LogisticRegression`.

`test-workflow.ipynb` also contains a later three-fold search and a pipeline placing SMOTE after preprocessing. That section explores a two-hidden-layer MLP with 200 and 150 neurons. It coexists with the earlier comparison and should not automatically be equated with the artifact saved earlier in the notebook.

Workflows are serialized as `new_workflow.joblib` and `new_workflow_lr.joblib`. The application loads the former. The exact correspondence between their internal parameters and a particular execution was not checked by deserializing the files during this review.

## 8. Inference and field tests

`inference.ipynb` loads the saved workflow, calculates classes and probabilities, and adds prediction columns to the dataset. It also identifies observations where the prediction differs from `Fallece`.

Its saved output shows confusion matrix `[[158, 18], [6, 26]]` for 208 observations. The report gives approximately 0.88 accuracy, recall of 0.90 for class 0 and 0.81 for class 1, and F1 of 0.93 and 0.68. The displayed AUC-ROC is 0.86, calculated from predicted classes. Because the notebook applies the model to the loaded dataset, these figures alone do not establish performance on patients independent of training.

`Prueba de campo.ipynb` evaluates an Excel field-test dataset. Its output preserves approximately 0.5982 balanced accuracy and 0.0862 Cohen's kappa. Recall by class is 0.3333 and 0.8630, with F1 of 0.1429 and 0.9130. Class encoding must be checked before directly comparing this with later workflows.

The newer workflows' saved internal test outputs use 21 observations, including three in class 1. The initial blocks for both MLP and logistic regression show class-1 recall of 1.00 and precision of 0.38. This represents three detected cases with several false positives, not a precise estimate of population sensitivity.

Their field-test outputs contain 76 observations: 73 in class 0 and three in class 1.

| Workflow | Accuracy | Class-1 precision | Class-1 recall | Class-1 F1 | Value labeled AUC-ROC |
| --- | --- | --- | --- | --- | --- |
| MLP | 0.82 | 0.18 | 1.00 | 0.30 | 0.90 |
| Logistic regression | 0.68 | 0.08 | 0.67 | 0.14 | 0.68 |

These are preserved outputs, not results reproduced for this document. Mortality prevalence and the small number of positive cases limit conclusions. Low precision also shows that detecting those cases can produce many alerts for patients who survive.

## 9. Prediction explainability

`Classification.ipynb` contains experiments with LIME, SHAP, Integrated Gradients, and Saliency. It uses both `interpret.blackbox` implementations and dedicated LIME and SHAP libraries. Gradient methods use Captum with PyTorch.

Part of the work copies coefficients and intercepts from a scikit-learn MLP into a PyTorch network. The architecture declared in that cell has six inputs, two hidden layers of 50 neurons with `tanh` activation, and a sigmoid output. This transfer allows gradient-based techniques to operate on a representation of the classifier.

The repository preserves serialized explainers in `Explainers/`, including Integrated Gradients, Saliency, LIME, and SHAP files. Local visualizations were generated to explore positive and negative feature contributions for individual observations.

A perturbation-based evaluation was also implemented. `modify_instance` changes features selected by relevance, and `calculate_fidelity` calculates the fraction of repetitions retaining the predicted class. This records stability under the notebook's defined perturbations. It does not by itself demonstrate clinical causality or universal explanation fidelity.

The current application uses `Explainers/ig_explainer.pkl`, while predictions come from `new_workflow.joblib`. This review did not verify that the explainer represents exactly the same model and preprocessing as the updated workflow. Their relationship remains to be checked before interpreting the attributions with confidence.

## 10. Streamlit web application

`web_app.py` implements an interface for estimating non-survival at ICU discharge. Its sidebar collects age, two admission diagnoses, the second discharge diagnosis, APACHE II, and ventilation duration.

The application includes 49 codes, numbered 0 through 48. Code 0 represents an empty option; the remaining entries include diagnoses and conditions such as sepsis, shock, severe traumatic brain injury, renal failure, and others. Dictionaries display descriptions while sending the expected numerical codes to the model.

The implemented flow is:

1. Enter patient characteristics.
2. Review an input table with English labels.
3. Select `Predict` to obtain `predict_proba(input_df)[:, 1]`.
4. View the percentage probability under a heading mentioning a 50% cutoff.
5. Select `Explain` to calculate attributions and request a narrative.

The probability is retained in `st.session_state`. Attempting to explain before predicting produces a request to make a prediction first. The relevance chart orders features by absolute value, using orange for positive contributions and blue for negative ones.

Execution depends on serialized files being available and compatible with installed libraries. This summary does not include a Streamlit startup test or interactive verification.

## 11. Language-model narratives

`llms/model-explain-text.ipynb` explores turning numerical explanations into text. It loads the pipeline, prepares SHAP explanations, and builds a prompt containing feature names, observed values, and contributions. History records this experiment before narrative generation was brought into the interface.

In the current `web_app.py`, text is built from Integrated Gradients attributions. Diagnosis codes are converted to descriptions before sending them to the generator. The prompt requests a clinical synthesis of the most influential factors in the context of hidden ICU mortality.

The key is read from `.env` using `GEMINI_API_KEY`. The code configures Gemini, uses `gemini-2.5-flash-lite`, and sets temperature to 0.7. The response must begin with `Narrative:`; the interface removes the prefix and displays the text in an information box.

This integration adds a textual presentation layer for attributions. The local pipeline still computes the probability. The code transmits clinical values and attributions included in the prompt to the provider; this review did not transmit them or evaluate the clinical correctness of generated narratives.

## 12. Artifacts and repository organization

The work is distributed among notebooks, models, explainers, and supporting files:

| File or group | Preserved contribution |
| --- | --- |
| `Classification.ipynb` | Exploration, classification, explainability, weight transfer, and perturbations. |
| `Classification_*` notebooks | Oversampling and model-selection variants. |
| `All.ipynb` | Auxiliary notebook importing explainability tools. |
| `Prueba de campo.ipynb` | Evaluation using field-test data in Excel. |
| `inference.ipynb` | Predictions, probabilities, and error review. |
| `test-*workflow.ipynb` notebooks | Categorical and numerical preprocessing, augmentation comparisons, and serialization. |
| `llms/model-explain-text.ipynb` | Narrative experiments based on explanations. |
| `Modelos/*.pkl` | Historical KNN, logistic regression, and MLP classifiers. |
| `Explainers/*.pkl` | Explainers saved using several libraries. |
| `modelo_completo.pth`, `modelo_mlp_ROS_pt.pth` | PyTorch artifacts. |
| `new_workflow.joblib`, `new_workflow_lr.joblib` | Later serialized workflows. |
| `Resultados_*.txt`, `configuracion_*.txt` | Experiment metrics and hyperparameters. |
| `web_app.py` | Streamlit interface and Gemini narrative integration. |
| `requirements.txt` | Environment dependencies. |
| `.devcontainer/devcontainer.json` | Development environment and Streamlit startup configuration. |

The repository also contains IDE configuration, Jupyter checkpoints, and remnants in `.tmp.driveupload/`. These are versioned materials, but they do not represent additional model capabilities.

## 13. Development environment and dependencies

The environment combines data analysis, machine learning, visualization, explainability, and text-generation tools. Declared dependencies include pandas, NumPy, scikit-learn, imbalanced-learn, PyTorch, Captum, LIME, SHAP, InterpretML, Matplotlib, Streamlit, joblib, cloudpickle, and language-service clients.

History shows several expansions and reductions of `requirements.txt`, plus adjustments to resolve conflicts and remove Windows-specific dependencies. The reviewed file pins many versions and is encoded in UTF-16. Experiment configuration files use another encoding, so not all text can be read directly as UTF-8.

The development container uses Python 3.11 on Debian Bullseye. It installs dependencies, prepares Streamlit, and configures `web_app.py` with port 8501 forwarding. This documents a development execution path, not an operational public deployment.

Ignoring `.env` keeps new local keys out of ordinary Git tracking. Tabular data files and a virtual-environment directory are also excluded. Full reproduction requires the external datasets referenced by the notebooks.

## 14. Established findings and remaining checks

The preserved work establishes a broad experimental sequence: clinical data inspection, three classifier families, several oversampling techniques, hyperparameter selection, internal and field evaluation, local explanations, and an interface with narrative generation.

The reviewed evidence leaves these checks open:

- Reproduce experiments using the original data and an unambiguous record of partitions.
- Separate fitting, oversampling, and evaluation rigorously, especially when interpreting older results.
- Verify the meaning and consistency of the positive class across dataset versions.
- Calculate AUC from probabilities when measuring discrimination and distinguish it from label-based evaluation.
- Confirm `TiempoVAM` units and the availability of discharge diagnoses at the intended prediction time.
- Verify that the application's explainer corresponds to the pipeline producing its probability.
- Associate each serialized model with its source data, parameters, dependency versions, and metrics.
- Evaluate sensitivity, precision, and false-alert volume using more positive cases.
- Assess probability calibration and threshold usefulness if results are intended to inform clinical decisions.

These are limitations and possible next steps, not work completed while preparing the summary. A prototype and saved metrics do not establish prospective clinical validation, hospital integration, or demonstrated clinical use.

## 15. Assessment of completed work

The repository records the progression from initial classification tests to a prototype that accepts clinical data, produces a probability, and provides a visual explanation with text. It preserves alternative models and several attempts to improve diagnosis handling, numerical scaling, and class imbalance treatment.

Its documented contributions include KNN, logistic regression, and MLP experiments; ROS and synthetic-method comparisons; field tests; four explainer families; and language-model narratives. The files make it possible to study both progress and differences between workflow generations.

The reviewed state is a research and prototyping project with preserved historical results. Further evaluation needs to connect those results to reproducible executions and verify that prediction, preprocessing, and explanation describe the same system.

## 16. Internal sources

Historical reconstruction uses available Git history from `bdba1c0` through `2cf8140`. Functional descriptions draw on `web_app.py`, the notebooks listed here, result and configuration files, `requirements.txt`, `.gitignore`, and `.devcontainer/devcontainer.json`.

The presence of versioned models and explainers was checked without deserializing them. Notebooks were inspected as JSON documents, including code and saved outputs, without executing cells. No external sources were consulted for the original experimental summary, and no patient-identifying data were included. Public SADUCI references were subsequently verified when documenting the integration.
