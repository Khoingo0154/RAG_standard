from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from api.main import app
from retrieval.models import RAGResponse
from retrieval.router import QueryIntent, QueryRouter
from services.football_api import FootballApiClient

client = TestClient(app)


def test_football_api_client_status_mocked():
    fb_client = FootballApiClient(api_key="fake-key")
    with patch.object(fb_client, "_request") as mock_req:
        mock_req.return_value = {
            "response": {
                "account": {"email": "test@gmail.com"},
                "subscription": {"plan": "Free", "active": True},
                "requests": {"current": 10, "limit_day": 100},
            }
        }

        status = fb_client.get_status()
        assert status["response"]["subscription"]["active"] is True
        assert status["response"]["requests"]["limit_day"] == 100


def test_football_api_client_search_teams_mocked():
    fb_client = FootballApiClient(api_key="fake-key")
    with patch.object(fb_client, "_request") as mock_req:
        mock_req.return_value = {
            "response": [
                {
                    "team": {"id": 42, "name": "Arsenal", "country": "England"},
                    "venue": {"name": "Emirates Stadium", "city": "London"},
                }
            ]
        }

        teams = fb_client.search_teams("Arsenal")
        assert len(teams) == 1
        assert teams[0]["team"]["name"] == "Arsenal"


def test_football_api_client_search_players_mocked():
    fb_client = FootballApiClient(api_key="fake-key")
    with patch.object(fb_client, "_request") as mock_req:
        mock_req.return_value = {
            "response": [
                {
                    "player": {"id": 154, "name": "L. Messi", "nationality": "Argentina"}
                }
            ]
        }

        players = fb_client.search_players("Messi")
        assert len(players) == 1
        assert players[0]["player"]["name"] == "L. Messi"


def test_football_api_client_fetch_data_for_query_team():
    fb_client = FootballApiClient(api_key="fake-key")
    with patch.object(fb_client, "search_teams") as mock_st, \
         patch.object(fb_client, "get_team_last_fixtures") as mock_fx:

        mock_st.return_value = [
            {
                "team": {"id": 42, "name": "Arsenal", "country": "England"},
                "venue": {"name": "Emirates Stadium", "city": "London"},
            }
        ]
        mock_fx.return_value = []

        res = fb_client.fetch_data_for_query("Thông tin về Arsenal")
        assert res["type"] == "team_info"
        assert res["team"]["name"] == "Arsenal"


def test_widgets_html_endpoint():
    response = client.get("/widgets")
    assert response.status_code == 200
    assert "api-sports-widget" in response.text
    assert "data-type=\"leagues\"" in response.text
    assert "data-type=\"config\"" in response.text


def test_football_status_endpoint():
    with patch("services.football_api.FootballApiClient.get_status") as mock_status:
        mock_status.return_value = {"response": {"subscription": {"plan": "Free"}}}
        response = client.get("/football/status")
        assert response.status_code == 200
        assert response.json()["response"]["subscription"]["plan"] == "Free"


def test_football_teams_endpoint():
    with patch("services.football_api.FootballApiClient.search_teams") as mock_st:
        mock_st.return_value = [{"team": {"id": 42, "name": "Arsenal"}}]
        response = client.get("/football/teams?search=Arsenal")
        assert response.status_code == 200
        data = response.json()
        assert data["results"] == 1
        assert data["teams"][0]["team"]["name"] == "Arsenal"


def test_football_players_endpoint():
    with patch("services.football_api.FootballApiClient.search_players") as mock_sp:
        mock_sp.return_value = [{"player": {"id": 154, "name": "L. Messi"}}]
        response = client.get("/football/players?search=Messi")
        assert response.status_code == 200
        data = response.json()
        assert data["results"] == 1
        assert data["players"][0]["player"]["name"] == "L. Messi"


@patch("google.genai.Client")
def test_router_classifies_football_live_data(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.models.generate_content.return_value = MagicMock(text="FOOTBALL_LIVE_DATA")

    router = QueryRouter(gemini_api_key="fake-key")
    intent = router.classify("Tỷ số trận Arsenal hôm nay thế nào?")
    assert intent == QueryIntent.FOOTBALL_LIVE_DATA


def test_pipeline_handles_football_live_data():
    from retrieval.pipeline import search_and_generate

    mock_router = MagicMock()
    mock_router.classify.return_value = QueryIntent.FOOTBALL_LIVE_DATA

    with patch("services.football_api.FootballApiClient.fetch_data_for_query") as mock_fetch, \
         patch("retrieval.generator.Generator._generate_gemini") as mock_gen:

        mock_fetch.return_value = {"type": "team_info", "team": {"name": "Arsenal"}}
        mock_gen.return_value = "Arsenal là câu lạc bộ bóng đá hàng đầu nước Anh, sân vận động Emirates."

        response = search_and_generate("Thông tin câu lạc bộ Arsenal", router=mock_router)

        assert response.intent == "FOOTBALL_LIVE_DATA"
        assert "Arsenal" in response.answer
        assert "API-Football" in response.answer
        assert response.sources == []
