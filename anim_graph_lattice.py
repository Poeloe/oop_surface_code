import plotly.graph_objects as go


class anim_2D(object):
    def __init__(self, graph, redraw=0, lw=3, sz=15, *args, **kwargs):
        self.graph = graph
        self.cl = self.get_rgb([0.8, 0.8, 0.8])  # Line color
        self.cc = self.get_rgb([0.7, 0.7, 0.7])  # Qubit color
        self.ec = self.get_rgb([0.3, 0.3, 0.3])  # Erasure color
        self.cx = self.get_rgb([0.9, 0.3, 0.3])  # X error color
        self.cz = self.get_rgb([0.5, 0.5, 0.9])  # Z error color
        self.cy = self.get_rgb([0.9, 0.9, 0.5])  # Y error color
        self.cX = self.get_rgb([0.9, 0.7, 0.3])  # X quasiparticle color
        self.cZ = self.get_rgb([0.3, 0.9, 0.3])  # Z quasiparticle color
        self.C1 = [self.cx, self.cz]
        self.C2 = [self.cX, self.cZ]

        self.scatter_size = sz
        self.lw = lw
        self.rd = redraw

        self.iter = 0
        self.init_plot()


    def get_rgb(self, L):
        def multiply(*args):
            return [int(i*255) for i in args]
        return "rgb({}, {}, {})".format(*multiply(*L))


    def init_plot(self):

        self.fig_dict = {
            "data"  : None,
            "layout": {"title": {"text": "test plot"}},
            "frames": []
        }
        self.fig_dict["layout"]["sliders"] = {
            "args": [
                "transition", {
                    "duration": 400,
                    "easing": "cubic-in-out"
                }
            ],
            "initialValue": "0",
            "plotlycommand": "animate",
            "values": [],
            "visible": True
        }
        self.fig_dict["layout"]["updatemenus"] = [
            {
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 500, "redraw": self.rd},
                                        "fromcurrent": True, "transition": {"duration": 300,
                                                                            "easing": "quadratic-in-out"}}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": self.rd},
                                          "mode": "immediate",
                                          "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }
        ]

        self.sliders_dict = {
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 20},
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 300, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": []
        }

        self.fig_dict["data"] = self.get_frame("Initial")


    def get_stab_layer(self, z=0):

        X, Y, Z, name, symbol, colors = [], [], [], [], [], []
        for vertex in self.graph.S[z].values():
            type, yb, xb = vertex.sID
            s = "cross" if type else "square"
            X.append(xb + .5*type)
            Y.append(yb + .5*type)
            Z.append(vertex.z)
            name.append(str(vertex))
            symbol.append(s)
            if vertex.parity:
                colors.append(self.C1[type])
            else:
                colors.append(self.cl)
        return X, Y, Z, name, symbol, colors

    def get_qubit_layer(self, z=0):
        X, Y, Z, name, cm, cl, op = [], [], [], [], [], [], []
        for qubit in self.graph.Q[z].values():
            (td, yb, xb) = qubit.qID
            x, y = (xb+.5, yb) if td == 0 else (xb, yb+.5)
            X.append(x)
            Y.append(y)
            Z.append(qubit.z)
            name.append(str(qubit))

            X_error = qubit.E[0].state
            Z_error = qubit.E[1].state

            if X_error and not Z_error:
                cl.append(self.cx)
            elif Z_error and not X_error:
                cl.append(self.cz)
            elif X_error and Z_error:
                cl.append(self.cy)
            else:
                cl.append(self.cc)

            X_error = qubit.E[0].matching
            Z_error = qubit.E[1].matching

            if X_error and not Z_error:
                cm.append(self.cx)
            elif Z_error and not X_error:
                cm.append(self.cz)
            elif X_error and Z_error:
                cm.append(self.cy)
            else:
                cm.append("white")

            if qubit.erasure:
                op.append(0.5)
            else:
                op.append(1)
        return X, Y, Z, name, cm, cl, op


    def get_frame(self, stepname=None):


        data = []

        X, Y, Z, name, symbol, colors = self.get_stab_layer(0)

        data.append({
                "x": X,
                "y": Y,
                "mode": "markers",
                "hoverinfo": "text",
                "text": name,
                "marker": {
                    "sizemode": "diameter",
                    "color": colors,
                    "size": self.scatter_size,
                    "symbol": symbol
                },
                "name": "stabs"
            })

        X, Y, Z, name, cm, cl, op = self.get_qubit_layer(0)

        data.append({
                "x": X,
                "y": Y,
                "mode": "markers",
                "hoverinfo": "text",
                "text": name,
                "marker": {
                    "sizemode": "diameter",
                    "color": cm,
                    "opacity": op,
                    "size": self.scatter_size,
                    "line": {
                        "width": self.lw,
                        "color": cl,
                    },
                },
                "name": "qubits"
            })

        frame = {"data": data, "name": self.iter}
        self.fig_dict["frames"].append(frame)
        self.slider(stepname)

        return data


    def slider(self, stepname=None):
        if stepname is None:
            stepname = self.iter
        slider_step = {"args": [
                [self.iter],
                {"frame": {"duration": 300, "redraw": self.rd},
                 "mode": "immediate",
                 "transition": {"duration": 300}}
            ],
                "label": stepname,
                "method": "animate"}

        self.sliders_dict["steps"].append(slider_step)
        self.iter += 1

    def plot_animation(self):

        self.fig_dict["layout"]["sliders"]["values"] = [i for i in range(self.iter)]
        self.fig_dict["layout"]["sliders"] = [self.sliders_dict]

        fig = go.Figure(self.fig_dict)
        fig.show()



class anim_3D(anim_2D):
    def __init__(self, graph, *args, **kwargs):
        super().__init__(graph, redraw=1, lw=5, sz=10, *args, **kwargs)

    def get_frame(self, stepname=None):

        data = []

        X, Y, Z, name, symbol, colors = [], [], [], [], [], []
        for z in self.graph.range:
            x, y, z, n, s, c = self.get_stab_layer(z)
            X.extend(x)
            Y.extend(y)
            Z.extend(z)
            name.extend(n)
            symbol.extend(s)
            colors.extend(c)

        data.append({
                "x": X,
                "y": Y,
                "z": Z,
                "type": "scatter3d",
                "mode": "markers",
                "hoverinfo": "text",
                "text": name,
                "marker": {
                    "sizemode": "diameter",
                    "color": colors,
                    "size": self.scatter_size,
                    "symbol": symbol
                },
                "name": "stabs"
            })

        X, Y, Z, name, cm, cl, op = [], [], [], [], [], [], []
        for z in self.graph.range:
            x, y, z, n, c1, c2, o = self.get_qubit_layer(z)
            X.extend(x)
            Y.extend(y)
            Z.extend(z)
            name.extend(n)
            cm.extend(c1)
            cl.extend(c2)
            op.extend(o)

        data.append({
                "x": X,
                "y": Y,
                "z": Z,
                "type": "scatter3d",
                "mode": "markers",
                "hoverinfo": "text",
                "text": name,
                "marker": {
                    "sizemode": "diameter",
                    "color": cm,
                    "opacity": 1,
                    "size": self.scatter_size,
                    "line": {
                        "width": self.lw,
                        "color": cl,
                    },
                },
                "name": "qubits"
            })

        frame = {"data": data, "name": self.iter}
        self.fig_dict["frames"].append(frame)
        self.slider(stepname)

        return data
