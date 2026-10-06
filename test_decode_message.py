"""Small offline checks for decode_message.py; run with python test_decode_message.py."""

import contextlib
import io
import unittest
from unittest.mock import patch

import requests

import decode_message


def fake_response(body: str, status: int = 200) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response.url = "https://example.test/doc"
    response._content = body.encode("utf-8")
    return response


class DecodeMessageTests(unittest.TestCase):
    def run_page(self, body: str, status: int = 200):
        output, errors = io.StringIO(), io.StringIO()
        with patch("decode_message.requests.get", return_value=fake_response(body, status)):
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                code = decode_message.main(["decode_message.py", "https://example.test/doc"])
        return code, output.getvalue(), errors.getvalue()

    def test_normal_table_and_malformed_row(self):
        html = """<table>
        <tr><td>x-coordinate</td><td>Character</td><td>y-coordinate</td></tr>
        <tr><td>0</td><td>A</td><td>0</td></tr>
        <tr><td>1</td><td>B</td><td>1</td></tr>
        <tr><td>bad</td><td>C</td><td>2</td></tr>
        </table>"""
        code, output, errors = self.run_page(html)
        self.assertEqual(code, 0)
        self.assertEqual(output, " B\nA\n")
        self.assertIn("row 4", errors)
        self.assertIn("Data rows parsed: 2; skipped: 1; grid: 2x2", errors)

    def test_http_404(self):
        code, _, errors = self.run_page("missing", 404)
        self.assertEqual(code, 1)
        self.assertIn("HTTP request failed with status 404", errors)

    def test_empty_page(self):
        code, _, errors = self.run_page("   ")
        self.assertEqual(code, 1)
        self.assertIn("response is empty", errors)

    def test_no_table(self):
        code, _, errors = self.run_page("<html><p>no table</p></html>")
        self.assertEqual(code, 1)
        self.assertIn("contains no table", errors)

    def test_every_data_row_invalid(self):
        html = """<table><tr><td>x-coordinate</td><td>Character</td><td>y-coordinate</td></tr>
        <tr><td>-1</td><td>A</td><td>0</td></tr></table>"""
        code, _, errors = self.run_page(html)
        self.assertEqual(code, 1)
        self.assertIn("no valid points", errors)


if __name__ == "__main__":
    unittest.main()
