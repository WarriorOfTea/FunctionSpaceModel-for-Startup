"""Короткий пример: один непрерывный план для трёх заданных случаев.

Запуск из корня проекта: uv run python examples/simple_continuous.py
Параметры вымышлены; расчёт не является прогнозом реального бизнеса.
"""

from pathlib import Path

from function_space_model.scenarios import standard_disturbances
from function_space_model.variational import VariationalConfig, optimize_continuous


def main():
    # Здесь меняются цель, срок, начальные деньги и предположения об откликах.
    config = VariationalConfig.from_file(Path(__file__).with_name("continuous.json"))
    events = standard_disturbances(config)

    # Одни цена и расходы; цель и резерв проверяются в каждом случае.
    report, _ = optimize_continuous(config, segments=6, disturbances=events)
    if report["plan"] is None:
        print("Проверенный путь не найден. Это не доказательство недостижимости.")
        return 2

    print("Условный пример, денежные суммы в условных единицах.")
    print(f"Цель: {config.target_profit:,.0f} прибыли за последние "
          f"{config.profit_window:g} г. к концу срока {config.years:g} г.")
    print(f"Штраф найденного пути J: {report['penalty']:.6f}")
    lower = report["analytic_lower_bound"]["value"]
    if lower is not None:
        print(f"Аналитическая нижняя граница J: {lower:.6f}")

    names = {"nominal": "Без сбоев",
             "development_delay_3_months": "Задержка развития на 3 месяца",
             "service_outage_1_month": "Простой на месяц"}
    checks = [report["nominal"], *report["scenario_checks"]]
    for check in checks:
        print(f"{names.get(check['name'], check['name'])}: "
              f"прибыль {check['window_profit']:,.0f}; "
              f"минимум денег {check['minimum_cash']:,.0f}")
    passed = sum(c["target_met"] and c["feasible_cash_and_capacity"] for c in checks)
    print(f"Цель, резерв и мощность: {passed}/{len(checks)} случаев.")
    print("Проверка численная, с допусками. Глобальный минимум не доказан.")
    print("Защита относится к отдельным случаям; их сочетание сюда не входит.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
