from bs4 import BeautifulSoup

days = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]

def Q1(file_path):  # DO NOT modify this line
    soup = BeautifulSoup(open(file_path), "html.parser")
    bud_day = soup.find_all("div", class_="bud-day-col")
    res = {day: 0 for day in days}
    for i in range(0, len(bud_day), 3):
        d = bud_day[i].text.split()[0].lstrip("วัน").rstrip("ที่")
        if "พฤหัส" in d:
            res["พฤหัสบดี"] += 1
        else:
            res[d] += 1
    return [res[day] for day in days]


def Q2(file_path):  # DO NOT modify this line
    soup = BeautifulSoup(open(file_path), "html.parser")
    vis_day = soup.find("a", attrs={'title':'วันวิสาขบูชา'})
    if vis_day is None:
        return None
    siblings = vis_day.parent.previous_siblings
    for sib in siblings:
        text = sib.text.strip()
        if not text:
            continue
        cond = text.split()[0].lstrip('วัน').rstrip('ที่')
        if "พฤหัส" in cond:
            return text
        if cond in days:
            return text
    return None

exec(input().strip())  # do not delete this line
