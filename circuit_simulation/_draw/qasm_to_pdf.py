import subprocess
import shutil
import os
import sys
sys.path.insert(1, os.path.abspath(os.getcwd()))
from circuit_simulation._draw.qasm2texLib import main as qasm2pdf
import warnings


def create_pdf_from_qasm(file_name, tex_file_name):
    pdf_file_name = tex_file_name.replace(".tex", ".pdf")
    destination_file_path = os.path.dirname(os.path.realpath(__file__))
    failed = False
    if not os.path.exists(pdf_file_name):
        with open(file_name, 'r') as qasm_file:
            qasm2pdf(qasm_file, tex_file_name)

        try:
            FNULL = open(os.devnull, 'w')
            proc = subprocess.Popen(['pdflatex', tex_file_name], stdout=FNULL)
            proc.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            failed = True
            warnings.warn("Failed to execute latex to pdf command! Timeout expired. No Latex file was created.")

        if not failed and not proc.returncode == 0:
            if os.path.exists(pdf_file_name):
                os.unlink(pdf_file_name)
            failed = True
            warnings.warn("Failed to execute latex to pdf command! No Latex file was created.")
        output_file_path = os.path.join(os.path.abspath(os.getcwd()), tex_file_name.split(os.sep)[-1])

        os.unlink(tex_file_name)
        os.unlink(output_file_path.replace(".tex", ".idx"))
        os.unlink(output_file_path.replace(".tex", ".aux"))
        os.unlink(output_file_path.replace(".tex", ".log"))
        if not os.path.exists(os.path.join(destination_file_path, "circuit_pdfs")):
            os.mkdir(os.path.join(destination_file_path, "circuit_pdfs"))
        if not failed:
            shutil.move(output_file_path.replace(".tex", ".pdf"),
                        os.path.join(destination_file_path,
                                     "circuit_pdfs",
                                     tex_file_name.split(os.sep)[-1].replace(".tex", ".pdf")))

    os.unlink(os.path.join(destination_file_path, tex_file_name.split(os.sep)[-1].replace(".tex", ".qasm")))
    if not failed:
        print("\nPlease open circuit pdf manually with file name: {}\n".format(pdf_file_name))

