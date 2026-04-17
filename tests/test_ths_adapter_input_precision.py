import unittest

from src.broker.ths_adapter import (
    _THS_AMOUNT_CONTROL_ID,
    _THS_PRICE_CONTROL_ID,
    _THS_SECURITY_CONTROL_ID,
    _normalize_trade_input_text,
)


class THSTradeInputPrecisionTests(unittest.TestCase):
    def test_stock_price_is_forced_to_two_decimals(self):
        self.assertEqual(
            _normalize_trade_input_text(_THS_PRICE_CONTROL_ID, "2.31232", "002219"),
            "2.31",
        )
        self.assertEqual(
            _normalize_trade_input_text(_THS_PRICE_CONTROL_ID, 2.3, "002219"),
            "2.30",
        )

    def test_convertible_bond_price_keeps_three_decimals(self):
        self.assertEqual(
            _normalize_trade_input_text(_THS_PRICE_CONTROL_ID, "123.45678", "113001"),
            "123.457",
        )

    def test_security_code_keeps_last_six_digits(self):
        self.assertEqual(
            _normalize_trade_input_text(_THS_SECURITY_CONTROL_ID, "sz002219"),
            "002219",
        )

    def test_amount_is_normalized_to_integer_text(self):
        self.assertEqual(
            _normalize_trade_input_text(_THS_AMOUNT_CONTROL_ID, "2000.0"),
            "2000",
        )


if __name__ == "__main__":
    unittest.main()
