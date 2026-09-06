"""Client tích hợp API-Football (API-Sports) để truy vấn tỷ số, lịch thi đấu, đội bóng và cầu thủ."""

import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

from shared.config import settings

logger = logging.getLogger(__name__)

# Cache ngắn hạn trong bộ nhớ để tránh tốn quota 100 requests/ngày của Free Tier
_CACHE: dict[str, tuple[float, Any]] = {}
CACHE_TTL_SECONDS = 120  # Cache 2 phút


class FootballApiClient:
    """Client giao tiếp với API-Football (v3.football.api-sports.io)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or settings.APISPORTS_KEY
        self.base_url = (base_url or settings.APISPORTS_BASE_URL).rstrip("/")

    def _request(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Gửi request HTTP GET đến API-Sports với xác thực header x-apisports-key."""
        if not self.api_key:
            return {"errors": ["Thiếu APISPORTS_KEY trong cấu hình."], "response": []}

        query_str = f"?{urllib.parse.urlencode(params)}" if params else ""
        url = f"{self.base_url}/{endpoint.lstrip('/')}{query_str}"

        # Kiểm tra cache
        import time
        now = time.time()
        if url in _CACHE:
            cached_time, cached_data = _CACHE[url]
            if now - cached_time < CACHE_TTL_SECONDS:
                return cached_data

        headers = {
            "x-apisports-key": self.api_key,
            "User-Agent": "RAG-Football-Assistant/1.0",
        }

        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                _CACHE[url] = (now, data)
                return data
        except Exception as e:
            logger.error(f"API-Sports request error ({url}): {e}")
            return {"errors": [str(e)], "response": []}

    def get_status(self) -> dict:
        """Kiểm tra trạng thái tài khoản và số lượt requests còn lại trong ngày."""
        return self._request("status")

    def get_live_fixtures(self) -> list[dict]:
        """Lấy danh sách các trận đấu đang diễn ra trực tiếp (Live scores)."""
        res = self._request("fixtures", {"live": "all"})
        return res.get("response", [])

    def search_teams(self, name: str) -> list[dict]:
        """Tìm kiếm thông tin câu lạc bộ / đội bóng theo tên."""
        if not name or len(name.strip()) < 3:
            return []
        res = self._request("teams", {"search": name.strip()})
        return res.get("response", [])

    def search_players(self, name: str) -> list[dict]:
        """Tìm kiếm hồ sơ cầu thủ (profile) theo tên."""
        if not name or len(name.strip()) < 3:
            return []
        res = self._request("players/profiles", {"search": name.strip()})
        return res.get("response", [])

    def get_team_last_fixtures(self, team_id: int, count: int = 5) -> list[dict]:
        """Lấy kết quả các trận đấu gần nhất của một đội bóng."""
        res = self._request("fixtures", {"team": str(team_id), "last": str(count)})
        return res.get("response", [])

    def get_standings(self, league_id: int = 39, season: int = 2024) -> list[dict]:
        """Lấy bảng xếp hạng giải đấu (Mặc định Ngoại Hạng Anh Premier League id=39)."""
        res = self._request("standings", {"league": str(league_id), "season": str(season)})
        return res.get("response", [])

    def fetch_data_for_query(self, query: str) -> dict:
        """Tự động phân tích câu hỏi người dùng để gọi endpoint phù hợp nhất từ API-Sports."""
        q_lower = query.lower()

        # 1. Câu hỏi về Tỷ số trực tiếp / Trận đang đá
        if any(w in q_lower for w in ["trực tiếp", "đang đá", "live", "tỷ số hôm nay", "kết quả hôm nay"]):
            live_matches = self.get_live_fixtures()
            return {
                "type": "live_fixtures",
                "count": len(live_matches),
                "data": [
                    {
                        "home": m["teams"]["home"]["name"],
                        "away": m["teams"]["away"]["name"],
                        "score": f"{m['goals']['home']} - {m['goals']['away']}",
                        "elapsed": m["fixture"]["status"]["elapsed"],
                        "status": m["fixture"]["status"]["long"],
                        "league": m["league"]["name"],
                    }
                    for m in live_matches[:10]
                ],
            }

        # 2. Câu hỏi về Cầu thủ cụ thể
        player_keywords = ["cầu thủ", "tiền đạo", "hậu vệ", "thủ môn", "tiền vệ", "sinh năm", "quốc tịch", "messi", "ronaldo", "mbappe", "haaland", "neymar"]
        if any(w in q_lower for w in player_keywords):
            # Trích xuất tên khả dĩ
            name = None
            for star in ["messi", "ronaldo", "mbappe", "haaland", "neymar", "debruyne", "salah", "lewandowski", "kane", "bellingham", "vinicius"]:
                if star in q_lower:
                    name = star
                    break

            if not name:
                # Tách từ sau chữ "cầu thủ"
                match = re.search(r"cầu thủ\s+([A-Za-zÀ-ỹ\s]+)", query, re.IGNORECASE)
                if match:
                    name = match.group(1).strip().split()[0]

            if name:
                players = self.search_players(name)
                if players:
                    p = players[0].get("player", {})
                    return {
                        "type": "player_profile",
                        "name": p.get("name"),
                        "firstname": p.get("firstname"),
                        "lastname": p.get("lastname"),
                        "age": p.get("age"),
                        "birth": p.get("birth", {}),
                        "nationality": p.get("nationality"),
                        "height": p.get("height"),
                        "weight": p.get("weight"),
                        "photo": p.get("photo"),
                    }

        # 3. Câu hỏi về Đội bóng / Tỷ số của một CLB
        famous_teams = {
            "arsenal": "Arsenal",
            "chelsea": "Chelsea",
            "liverpool": "Liverpool",
            "manchester united": "Manchester United",
            "mu": "Manchester United",
            "manchester city": "Manchester City",
            "man city": "Manchester City",
            "real madrid": "Real Madrid",
            "barcelona": "Barcelona",
            "bayern": "Bayern Munich",
            "psg": "Paris Saint Germain",
            "juventus": "Juventus",
            "milan": "AC Milan",
            "inter": "Inter",
            "tottenham": "Tottenham",
        }

        matched_team = None
        for key_alias, full_name in famous_teams.items():
            if key_alias in q_lower:
                matched_team = full_name
                break

        if matched_team:
            teams = self.search_teams(matched_team)
            if teams:
                t_obj = teams[0].get("team", {})
                v_obj = teams[0].get("venue", {})
                team_id = t_obj.get("id")

                # Lấy thêm các trận gần nhất nếu có team_id
                last_matches = []
                if team_id:
                    fx_data = self.get_team_last_fixtures(team_id, count=3)
                    for f in fx_data:
                        last_matches.append({
                            "date": f["fixture"]["date"][:10],
                            "home": f["teams"]["home"]["name"],
                            "away": f["teams"]["away"]["name"],
                            "score": f"{f['goals']['home']} - {f['goals']['away']}",
                            "status": f["fixture"]["status"]["short"],
                        })

                return {
                    "type": "team_info",
                    "team": {
                        "id": team_id,
                        "name": t_obj.get("name"),
                        "country": t_obj.get("country"),
                        "founded": t_obj.get("founded"),
                        "logo": t_obj.get("logo"),
                        "stadium": v_obj.get("name"),
                        "city": v_obj.get("city"),
                        "capacity": v_obj.get("capacity"),
                    },
                    "recent_matches": last_matches,
                }

        # 4. Fallback: Nếu không phát hiện cụ thể, kiểm tra các trận live đang diễn ra
        live_matches = self.get_live_fixtures()
        return {
            "type": "general_football_data",
            "live_matches_count": len(live_matches),
            "note": "Không phát hiện tên đội hoặc cầu thủ cụ thể, sẵn sàng tra cứu thông tin theo yêu cầu.",
        }
