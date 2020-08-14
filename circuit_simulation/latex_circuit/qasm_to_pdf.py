import os
import subprocess
import shutil
import os
import sys
sys.path.insert(1, os.path.abspath(os.getcwd()))
from circuit_simulation.latex_circuit.qasm2texLib import main as qasm2pdf


def create_pdf_from_qasm(file_name, tex_file_name):
    pdf_file_name = tex_file_name.replace(".tex", ".pdf")
    destination_file_path = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(pdf_file_name):
        with open(file_name, 'r') as qasm_file:
            qasm2pdf(qasm_file, tex_file_name)

        FNULL = open(os.devnull, 'w')
        proc = subprocess.Popen(['pdflatex', tex_file_name], stdout=FNULL)
        proc.communicate()

        retcode = proc.returncode
        if not retcode == 0:
            os.unlink(pdf_file_name)
            raise ValueError("Failed to execute latex to pdf command!")
        output_file_path = os.path.join(os.path.abspath(os.getcwd()), tex_file_name.split(os.sep)[-1])

        os.unlink(tex_file_name)
        os.unlink(output_file_path.replace(".tex", ".idx"))
        os.unlink(output_file_path.replace(".tex", ".aux"))
        os.unlink(output_file_path.replace(".tex", ".log"))
        shutil.move(output_file_path.replace(".tex", ".pdf"),
                    os.path.join(destination_file_path,
                                 "circuit_pdfs",
                                 tex_file_name.split("/")[-1].replace(".tex", ".pdf")))

    os.unlink(os.path.join(destination_file_path, tex_file_name.split(os.sep)[-1].replace(".tex", ".qasm")))
    print("\nPlease open circuit pdf manually with file name: {}\n".format(pdf_file_name))

