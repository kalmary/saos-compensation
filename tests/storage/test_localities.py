
import pytest

from dataset.clients import api


@pytest.mark.parametrize("debug", [False, True])
def test_locality_download_is_retained_only_for_debug(
    locality_database,
    monkeypatch,
    tmp_path,
    debug,
):
    """
    Verify locality download is retained only for debug.

    args:
        locality_database: Real temporary locality snapshot.
        monkeypatch: Scoped external HTTP replacement.
        tmp_path: Isolated temporary directory.
        debug: Whether to save diagnostic artifacts.

    returns:
        None.
    """
    from dataset.storage import localities

    content = locality_database.read_bytes()

    class Response:
        def __enter__(self):
            """
            Provide the   enter   behavior used by this operation.

            args:
                None

            returns:
                Result of this operation.
            """
            self.content = content
            return self

        def __exit__(self, *args):
            """
            Provide the   exit   behavior used by this operation.

            args:
                args: Input for this operation.

            returns:
                Result of this operation.
            """
            pass

        def raise_for_status(self):
            """
            Provide the raise for status behavior used by this operation.

            args:
                None

            returns:
                Result of this operation.
            """
            pass

    monkeypatch.setattr(api.requests, "get", lambda *args, **kwargs: Response())
    directory = tmp_path / "debug"
    rows, metadata = localities.load_localities(directory if debug else None)
    assert rows[0]["NAZWA"] == "Pisz"
    assert metadata["population_year"] is None
    assert directory.exists() == debug

    if debug:
        monkeypatch.setattr(
            api.requests, "get", lambda *args, **kwargs: pytest.fail("Unexpected download")
        )
        assert localities.load_localities(directory) == (rows, metadata)
        (directory / "OT_ADMS_P.gpkg").write_bytes(b"damaged snapshot")
        with pytest.raises(ValueError, match="integrity"):
            localities.load_localities(directory)
