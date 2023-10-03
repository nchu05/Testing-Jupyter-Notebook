import nbformat
import ast

# Load the Jupyter Notebook
notebook_path = 'CurveFitting.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as notebook_file:
    notebook_content = nbformat.read(notebook_file, as_version=4)

# Function to traverse and print the AST of code cells
def process_code_cells(cells):
    counter = 1
    for cell in cells:
        if cell.cell_type == 'code':
            code = cell.source
            try:
                parsed_code = ast.parse(code)
                # You can now traverse the AST and do whatever you need
                print("######### CODE CELL",counter,"#########")
                print(ast.dump(parsed_code, indent=4))
                counter +=1
            except SyntaxError as e:
                print(f"Error in code cell: {e}")

# Traverse and process the code cells
process_code_cells(notebook_content.cells)
