# WiDS Datathon Project


## Description

Neuropsychiatric disorders that occur in development, like anxiety,
depression, autism, and attention deficit hyperactivity disorder, or
ADHD, often differ in how and to what extent they affect males and
females. ADHD occurs in about 11% of adolescents, with around 14% of
boys and 8% of girls having a diagnosis. There is some evidence that
girls with ADHD can often go undiagnosed, as they tend to have more
inattentive symptoms which are harder to detect. Girls with ADHD who are
undiagnosed with continue suffering from symptoms that burden their
mental health and capacity to function.

Fundamental Question: What brain activity patterns are associated with ADHD; are they different between males and females, and if so, how?

## Notes:
- build a multi-outcome model to predict both an individual's sex and ADHD diagnosis using functional brain imaging data of adolescents and their socio-demographic, emotions, and parenting information
- double-weight females diagnosed with ADHD in the model
- optimize weighted F1-score

## Methods:
- a basic random forest model trained on the original data was used as a baseline; F1-Score: 0.3628
- after basic preprocessing (OHE), a neural network model yielded an F1-score of 0.4379
- imputation methods were implemented to prepare the quantitative and categorical data for other modeling techniques
- the imputed data was used in a random forest model; sample weights were used to double-weight females with ADHD and determine optimal threshold values for each response variable, which yielded an F1-score of 0.5553
- the imputed data was used in a support vector machine model; sample weights were used to double-weight females with ADHD and determine optimal threshold values for each response variable, which yielded an F1-score of 0.6015

## Findings:
- the highest F1-score obtained using the methods outlined was 0.6015 via the support vector machine model

## Project Organization

- `/code` Scripts
- `/data` Real data and predictive results
- `/presentations` Presentation slides.
- `.gitignore` Hidden Git instructions file.
- `.python-version` Hidden Python version for the reproducible
  environment.
- `requirements.txt` Information on the reproducible environment.
