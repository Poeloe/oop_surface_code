import plotly.graph_objects as go
import anim_graph_lattice as agl


class anim_2D(agl.anim_2D):

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
                        "args": [None, {"frame": {"duration": 500, "redraw": False},
                                        "fromcurrent": True, "transition": {"duration": 300,
                                                                            "easing": "quadratic-in-out"}}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": False},
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


    def get_frame(self, stepname=None):

        data = []

        def get_line(qubit, id, X, Y):

            edge = qubit.E[id]
            c1, c2 = self.C1[id], self.C2[id]
            name = str(edge)

            if edge.peeled or edge.support == 0:
                color, dash = self.cl, "solid"
            if edge.support == 1:
                color, dash = c1, "dot"
            if edge.support == 2 and edge.matching == 0:
                color, dash = c1, "solid"
            if edge.support == 2 and edge.matching == 1:
                color, dash = c2, "solid"

            return {
                    "x": X,
                    "y": Y,
                    "mode": "lines",
                    "hoverinfo": "text",
                    "text": name,
                    "line": {
                        "width": self.lw,
                        "color": color,
                        "dash": dash,
                    },
                }
        for qubit in self.graph.Q[0].values():
            (td, yb, xb) = qubit.qID
            x, y = (xb+.5, yb) if td == 0 else (xb, yb+.5)

            if td == 0:
                Y0, X0 = [y, y], [x-.5, x+.5]
                Y1, X1 = [y-.5, y+.5], [x, x]
            else:
                Y1, X1 = [y, y], [x-.5, x+.5]
                Y0, X0 = [y-.5, y+.5], [x, x]

            data.append(get_line(qubit, 0, X0, Y0))
            data.append(get_line(qubit, 1, X1, Y1))



        X, Y, name, cm, cl = [], [], [], [], []
        for vertex in self.graph.S[0].values():
            type, yb, xb = vertex.sID
            if vertex.state:
                cm.append(self.C2[type])
                cl.append(self.C1[type])
                X.append(xb + .5*type)
                Y.append(yb + .5*type)
                if vertex.cluster:
                    name.append("{}\n{}".format(vertex, vertex.cluster.parent))
                else:
                    name.append("{}".format(vertex))


        data.append({
                "x": X,
                "y": Y,
                "mode": "markers",
                "hoverinfo": "text",
                "text": name,
                "marker": {
                    "sizemode": "diameter",
                    "color": cm,
                    "size": self.scatter_size,
                    "line": {
                        "width": self.lw,
                        "color": cl,
                    },
                },
                "name": "stabs"
            })


        frame = {"data": data, "name": self.iter}
        self.fig_dict["frames"].append(frame)

        if stepname is None:
            stepname = self.iter

        slider_step = {"args": [
                [self.iter],
                {"frame": {"duration": 300, "redraw": False},
                 "mode": "immediate",
                 "transition": {"duration": 300}}
            ],
                "label": stepname,
                "method": "animate"}

        self.sliders_dict["steps"].append(slider_step)

        self.iter += 1

        return data



    def plot_animation(self):

        self.fig_dict["layout"]["sliders"]["values"] = [i for i in range(self.iter)]
        self.fig_dict["layout"]["sliders"] = [self.sliders_dict]

        fig = go.Figure(self.fig_dict)
        fig.show()
