# noinspection PyProtectedMember
class Qubit:

    def __init__(self, qc, index, qubit_type, waiting_time_idle=0, waiting_time_lde=0):
        self._index = index
        self._qubit_type = qubit_type
        self._waiting_time_idle = waiting_time_idle
        self._waiting_time_lde = waiting_time_lde
        self._qc = qc

    @property
    def index(self):
        return self._index

    @property
    def qubit_type(self):
        return self._qubit_type

    @property
    def waiting_time_idle(self):
        return self._waiting_time_idle

    @property
    def waiting_time_lde(self):
        return self._waiting_time_lde

    @property
    def density_matrix(self):
        return self._qc._qubit_density_matrix_lookup[self.index][0]

    def increase_waiting_time(self, amount, waiting_type='idle'):
        if waiting_type not in ['idle', 'LDE']:
            raise ValueError("Waiting type should be either 'idle' or 'LDE'")

        if waiting_type == 'idle':
            self._waiting_time_idle += amount
        elif waiting_type == 'LDE':
            self._waiting_time_idle += amount
            self._waiting_time_lde += amount

    def reset_waiting_time(self):
        self._waiting_time_idle = 0
        self._waiting_time_lde = 0