import ast
import os
import yaml
from yaml.loader import SafeLoader

def automateConvert(project):
    for dirpath, dirnames, filenames in os.walk(project):
        for file in filenames:
            filePath = os.path.join(dirpath, file)
            with open(filePath, "r") as current:
                

def copyDependencies(project):
    for dirpath, dirnames, filenames in os.walk(project):
        j = 0
        for file in filenames:
            filePath = os.path.join(dirpath, file)
            with open(filePath, "r") as current:
                doc = yaml.safe_load(current)
                for i in doc["dependencies"]:
                    with open("tests/requirements{}.txt".format(j), "a") as f:
                        if type(i) == str:
                            f.write(i +'\n')
                        else:
                            for d in i:
                                if d != "pip":
                                    f.write(d +'\n')
                j=j+1
                
def automateDownload(folder):
    for dirpath, dirnames, filenames in os.walk(folder):
        i = 0
        for file in filenames:
            filePath = os.path.join(dirpath, file)
            with open(filePath, "r") as current:
                os.chdir(filePath)
                os.system('cmd /k pip download -r requirements{}.txt'.format(j))
            i = i + 1        


def main():
    #copyDependencies("yml")
    automateDownload()

if __name__ == "__main__":
    main()