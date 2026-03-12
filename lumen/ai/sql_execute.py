from __future__ import annotations

import param

from panel.viewable import Viewer
from panel_material_ui import IconButton, Row


class SQLExecute(Viewer):

    view = param.Parameter()

    right = param.String(default="8px")

    top = param.String(default="0px")

    z_index = param.String(default="1000")

    icon_size = param.String(default="1.2em")

    def __init__(self, **params):
        super().__init__(**params)

        def execute_sql(event):
            editor = getattr(self.view, "_editor", None)
            spec = getattr(self.view, "spec", None)
            if editor is None:
                return
            if editor.value == spec:
                return
            try:
                editor.param.trigger("value")
            except Exception:
                self.view.spec = editor.value

        button = IconButton(
            icon="play_arrow",
            description="Execute SQL",
            size="small",
            color="primary",
            icon_size=self.icon_size,
            on_click=execute_sql,
            margin=0,
        )

        self._row = Row(
            button,
            styles={
                "position": "absolute",
                "top": self.top,
                "right": self.right,
                "z-index": self.z_index,
            },
        )

    def __panel__(self):
        return self._row

