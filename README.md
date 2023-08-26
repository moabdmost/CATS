# CATS: Comprehensive Automatic Text Scorer.
## Work in progress ...

Welcome to the Comprehensive Automatic Text Scorer repository! This application was originally developed for the University of Kuala Lumpur, but it's now open to the public. This repository contains the code and resources for the application that leverages advanced AI technologies to enhance the efficiency and accuracy of essay scoring, detect AI-generated text, and perform plagiarism checks.

## Features

- **Automatic Essay Scoring:** Utilizes the EXPATS Framework to provide automated essay scoring, enabling educators to assess a large number of essays efficiently.

- **AI-written Text Detection:** Built upon OpenAI's GPT-2 detector architecture, this feature identifies AI-generated content in essays, ensuring the authenticity and integrity of student submissions.

- **Plagiarism Checking:** Our plagiarism checker employs simple algorithms to identify instances of plagiarism and provide detailed reports to educators.

- **User Interface:** A user-friendly web interface has been developed from scratch, making it easy for educators to interact with the application. The UI is designed for in-class testing and provides a seamless experience.

## Requirements

[Python](https://www.python.org/?downloads) and [Poetry](https://python-poetry.org/) are required to run and install the dependencies.
```
apt-get install python-pip
pip install poetry

```

## Usage

To use the application, follow these steps:

1. Clone the repository to your local machine.
```
git clone https://github.com/your-username/https://github.com/moabdmost/AES_UniKL.git
```
2. Change the main folder name to `workspace`
3. Install the required dependencies for each feature:

```
cd workspace
pip install -r requirements.txt
```

4. Install EXPATS python dependencies via poetry.

```
cd expats
poetry install
poetry shell
```
5. [Download the detector and scorer models](https://drive.google.com/drive/folders/11cfavr7XmtXuxdVGSGFHx1r4ihvLWm6q?usp=sharing) and save the folder `models` in `workspace`.
6. Launch the web user interface by running the provided script:
```
python workspace/flask-aes-v2/app.py
```

7. Access the application in your web browser by navigating to `http://localhost:5000`.

## Contributions

We welcome contributions from the community to improve and expand the functionality of the application. If you have ideas, bug fixes, or new features to add, please feel free to fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE.txt).

## Contact

If you have any questions, suggestions, or feedback, you can reach us at [Megat Azmi](mailto:megatnorulazmi@unikl.edu.my) and [Mohamed Mostafa](mailto:mocshamohamed@gmail.com).

---

**Disclaimer:** This application was developed as part of a project for the University of Kuala Lumpur. The application's efficacy and accuracy in real-world scenarios may vary. Users are advised to exercise caution and judgment when using the application's results for important assessments and decisions.
