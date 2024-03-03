# Project SmartHealth

## Introduction
Artificial Intelligence (AI) is revolutionizing healthcare across diagnostic precision, treatment optimization, patient-centric care, operational efficiency, and research and development. Through advanced algorithms, AI analyzes vast datasets for early disease detection, tailors personalized treatment plans, and enhances patient engagement through virtual assistants and remote monitoring devices. Moreover, AI streamlines administrative tasks, predicts resource demands, and accelerates drug discovery, all while upholding ethical standards and ensuring patient privacy. As AI continues to evolve, its integration promises to elevate healthcare delivery, fostering a future where every individual receives optimized, efficient, and equitable care.

## Role And Application of AI in Helathcare

The project, *SmartHealth*, employs machine learning to predict the onset of various health conditions, facilitating early intervention. By analyzing comprehensive patient data, including electronic health records, lifestyle information, and genetic factors, our model identifies subtle patterns indicative of potential health risks. The system aims to provide personalized risk assessments, equipping healthcare professionals with valuable insights for proactive and targeted patient care. *SmartHealth* prioritizes transparency, interpretability, and seamless integration with existing healthcare systems for widespread impact.


These projects contain both the `.py` and `.ipynb` files of the projects. The datasets are availabe in the `data` folder.

### Pima Diabetes
This project implements a PyTorch deep Diabetic model for predicting the presence or absence of diabetes. A number of eight(8) features were used for the predicting if an individual has diabtes.
The features are `Pregnancies	Glucose	BloodPressure SkinThickness Insulin BMI DiabetesPedigreeFunction Age`. The pedicted `Output` is either `1 (presence of diabetes)` or `0 (absence of diabetes).`

The `dataset` used for this model is the [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database). This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.
The datasets consists of several medical predictor variables and one target variable, `Outcome`. Predictor variables includes the number of pregnancies the patient has had, their `BMI, insulin level, age`, and so on.

*Acknowledgements*
`Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988)`. [Using the ADAP learning algorithm to forecast the onset of diabetes mellitus](http://rexa.info/paper/04587c10a7c92baa01948f71f2513d5928fe8e81). In _Proceedings of the Symposium on Computer Applications and Medical Care_ (pp. 261--265). IEEE Computer Society Press.

*The Inspiration*
Can you build a machine learning model to accurately predict whether or not the patients in the dataset have diabetes or not?

### Settings for pima-diabetes
Clone the repo and run:
`pip install -q -r requirements.txt`


### Anatomy RAG
This prject implements an Retrieval Augmented Generation (RAG) for Anatomy and physiology using Medical Large Language model called `meditron.` The aim of the project is to enable answer anatomy and physiology related questions.

The Model model for embedding is [pubmedbert-base-embeddings](https://huggingface.co/NeuML/pubmedbert-base-embeddings) that is available in in Huggingface registry. The model for chat generation is [meditron-7B-chat-GGUF](https://huggingface.co/TheBloke/meditron-7B-GGUF). 

The `dataset` that was is the [Essentials of Anatomy and Physiology](https://repository.poltekkes-kaltim.ac.id/1144/1/Essentials%20of%20Anatomy%20and%20Physiology%20(%20PDFDrive%20).pdf) by `Valerie C. Scanlon, PhD` and `Tina Sanders`. This LLM RAG can answer Anatomy and Physiology related questions, to enable clinicians answer basic questions.

Chroma was used as the vector database for this project.

### Settings for Anatomy RAG
Clone the repo and run:
```bash
pip install -q -r requirements.txt



