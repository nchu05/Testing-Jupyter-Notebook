import os
import shutil

def automateConvertToPy(project):
    for dirpath, dirnames, filenames in os.walk(project):
        for file in filenames:
            filePath = os.path.join(dirpath, file)
            os.chdir(project)
            os.system('jupyter nbconvert --to python {}'.format(file))

def convertToIpynb(project):
    for dirpath, dirnames, filenames in os.walk(project):
        for file in filenames:
            filePath = os.path.join(dirpath, file)
            if filePath[-5:] == "ipynb":
                shutil.copy(filePath, "Qiskit/tutorials/algorithms_copy")
            if filePath[-2:] == "py":
                os.system('p2j -o {}'.format(filePath))

def checkSame(project, copy_folder):
    for dirpath, dirnames, filenames in os.walk(project):
        for file in filenames:
            filePath = os.path.join(dirpath, file)
            

def main():
    # automateConvertToPy("/Users/nathanchu/Documents/Testing-Jupyter-Notebook/Qiskit/tutorials/algorithms")
    # automateConvertToPy("/Users/nathanchu/Documents/Testing-Jupyter-Notebook/Qiskit/tutorials/circuits")
    # automateConvertToPy("/Users/nathanchu/Documents/Testing-Jupyter-Notebook/Qiskit/tutorials/circuits_advanced")
    # automateConvertToPy("/Users/nathanchu/Documents/Testing-Jupyter-Notebook/Qiskit/tutorials/operators")
    convertToIpynb("/Users/nathanchu/Documents/Testing-Jupyter-Notebook/Qiskit/tutorials/algorithms")
    


if __name__ == "__main__":
    main()