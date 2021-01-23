import unittest
import os
import pickle
import pandas as pd
from circuit_simulation.stabilizer_measurement_protocols.argument_parsing import compose_parser, group_arguments
from circuit_simulation.stabilizer_measurement_protocols.run_protocols import run_for_arguments, \
    additional_parsing_of_arguments
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
FILE_NAME = os.path.join(FILE_DIR, 'test_csv')


def prepare_arguments(file):
    parser = compose_parser()
    args = vars(parser.parse_args(['--argument_file', file]))
    args['csv_filename'] = FILE_NAME
    args = additional_parsing_of_arguments(**args)

    grouped_arguments = group_arguments(parser, **args)
    return args, grouped_arguments


class StabilizerProtocolSanityTest(unittest.TestCase):
    @classmethod
    def tearDown(cls) -> None:
        """ Deletes saved csv and pickle files after each test """
        for file in os.listdir(FILE_DIR):
            if file.endswith(".csv") or file.endswith('.pkl'):
                os.remove(os.path.join(FILE_DIR, file))

    def test_expedient_sanity(self):
        """
            Checks whether EXPEDIENT protocol has the superoperator values according to values found by Naomi Nickerson
        """
        args, grouped_args = prepare_arguments(os.path.join(FILE_DIR, 'UNIT_TEST_arguments.txt'))
        grouped_args[2]['protocol'] = ["expedient"]
        grouped_args[2]['pg'] = [0.006]

        grouped_args[1]['probabilistic'] = False

        filename = run_for_arguments(*grouped_args, **args)[0]
        df = pd.read_csv(filename + ".csv", sep=";", float_precision='round_trip', index_col=[0, 1])
        self.assertAlmostEqual(df.iloc[0, 0], 0.9117332641182799)

    def test_stringent_sanity(self):
        """
            Checks whether STRINGENT protocol has the superoperator values according to values found by Naomi Nickerson
        """
        args, grouped_args = prepare_arguments(os.path.join(FILE_DIR, 'UNIT_TEST_arguments.txt'))
        grouped_args[2]['protocol'] = ["stringent"]
        grouped_args[2]['pg'] = [0.0075]

        grouped_args[1]['probabilistic'] = False

        filename = run_for_arguments(*grouped_args, **args)[0]
        df = pd.read_csv(filename + ".csv", sep=";", float_precision='round_trip', index_col=[0, 1])
        self.assertAlmostEqual(df.iloc[0, 0], 0.9280778902420826)

    def test_swap_sanity(self):
        """
            Checks whether the swapped protocol version gives the same stabilizer fidelity when swap gate error is
            set to 0.
        """
        # ADD PROTOCOL TO LIST IF SWAP SANITY NEEDS TO BE CHECKED
        protocols = ["expedient", "stringent", "plain"]

        filenames = {True: [], False: []}

        for swap_value in [False, True]:
            args, grouped_args = prepare_arguments(os.path.join(FILE_DIR, 'UNIT_TEST_arguments.txt'))
            grouped_args[2]['protocol'] = protocols
            grouped_args[2]['pg'] = [0.001]

            grouped_args[1]['noiseless_swap'] = True
            grouped_args[1]['probabilistic'] = False
            grouped_args[1]['use_swap_gates'] = swap_value

            filenames[swap_value].extend(run_for_arguments(*grouped_args, **args))

        for file_swap, file in zip(*filenames.values()):
            df_swap = pd.read_csv(file_swap + ".csv", sep=";", float_precision='round_trip', index_col=[0, 1])
            df = pd.read_csv(file + ".csv", sep=";", float_precision='round_trip', index_col=[0, 1])
            self.assertAlmostEqual(df_swap.iloc[0, 0], df.iloc[0, 0])

    def test_probabilistic_sanity(self):
        """
            Check whether the fidelities of the probabilistic protocol are equal to the fidelity of the
            non-probabilistic protocol (this should be the case when there is no decoherence and no difference in
            error rate between measurement outcomes)
        """
        # ADD PROTOCOL TO LIST IF SWAP SANITY NEEDS TO BE CHECKED
        protocols = ["plain", "expedient", "stringent", "dyn_prot_4_4_1", "dyn_prot_4_14_1"]

        filenames = {True: [], False: []}

        for prob_value in [False, True]:
            args, grouped_args = prepare_arguments(os.path.join(FILE_DIR, 'UNIT_TEST_arguments.txt'))
            grouped_args[2]['protocol'] = protocols
            grouped_args[2]['pg'] = [0.001]

            # USES SWAPPED VERSION OF PROTOCOL
            grouped_args[1]['use_swap_gates'] = True
            grouped_args[1]['iterations'] = 1 if not prob_value else 20
            grouped_args[1]['probabilistic'] = prob_value

            filenames[prob_value].extend(run_for_arguments(*grouped_args, **args))

        for file_prob, file in zip(*filenames.values()):
            pkl_prob = pickle.load(open(file_prob + ".pkl", 'rb'))
            pkl = pickle.load(open(file + ".pkl", 'rb'))
            ghz_fid = pkl['ghz_fid'][0]
            stab_fid = pkl['stab_fid'][0]

            [self.assertAlmostEqual(ghz_fid, ghz_fid_prob) for ghz_fid_prob in pkl_prob['ghz_fid']]
            [self.assertAlmostEqual(stab_fid, stab_fid_prob) for stab_fid_prob in pkl_prob['stab_fid']]


if __name__ == "__main__":
    unittest.main()
