import re


def parse_table(table_name, table_lines):
    rows = []
    line_items = []

    for line in table_lines:

        line = line.strip()

        if not line:
            continue

        m = re.match(
            r"^(.*?)(\(?-?[\d,]+\)?(?:\s+\(?-?[\d,]+\)?)*)$",
            line
        )

        if not m:
            continue

        label = m.group(1).strip()

        values = re.findall(r"\(?-?[\d,]+\)?", m.group(2))

        rows.append(
            {
                "label": label,
                "values": values
            }
        )

        line_items.append(label)

    return {
        "table_name": table_name,
        "rows": rows,
        "line_items": line_items
    }