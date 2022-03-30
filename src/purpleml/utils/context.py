import pathlib


class Context:

    def __init__(self, name=None, out_dir="../_out"):
        self.name = name
        self.out_dir = out_dir

    def file(self, filename, folder=None, name=None):
        return context_path(
            filename, folder=folder, context_name=self.name if name is None else name, out_dir=self.out_dir)

    def folder(self, folder, name=None):
        return context_path(
            folder=folder, context_name=self.name if name is None else name, out_dir=self.out_dir)


def context_path(filename=None, folder=None, context_name=None, out_dir="../_out"):

    p = pathlib.Path(out_dir)

    if context_name is not None:
        p /= context_name

    if folder is not None:
        p /= folder

    if not p.exists():
        p.mkdir(parents=True)

    if filename is not None:
        p /= filename

    return p
