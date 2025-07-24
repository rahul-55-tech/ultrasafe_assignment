import json
import os
from threading import Lock
from typing import Any, Dict, List


class JSONDatabase:
    def __init__(self, db_folder="db"):
        self.db_folder = db_folder
        if not os.path.exists(self.db_folder):
            os.makedirs(self.db_folder)
        self.locks = {}

    def _get_lock(self, table_name: str) -> Lock:
        if table_name not in self.locks:
            self.locks[table_name] = Lock()
        return self.locks[table_name]

    def _get_table_path(self, table_name: str) -> str:
        return os.path.join(self.db_folder, f"{table_name}.json")

    def read_table(self, table_name: str) -> List[Dict[str, Any]]:
        table_path = self._get_table_path(table_name)
        if not os.path.exists(table_path):
            return []
        with self._get_lock(table_name):
            with open(table_path, "r") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return []

    def write_table(self, table_name: str, data: List[Dict[str, Any]]):
        table_path = self._get_table_path(table_name)
        with self._get_lock(table_name):
            with open(table_path, "w") as f:
                json.dump(data, f, indent=4, default=str)

    def insert(self, table_name: str, item: Dict[str, Any]) -> Dict[str, Any]:
        data = self.read_table(table_name)
        # Simple auto-incrementing ID
        new_id = (max([d.get("id", 0) for d in data]) if data else 0) + 1
        item["id"] = new_id
        data.append(item)
        self.write_table(table_name, data)
        return item

    def find_one(self, table_name: str, conditions: Dict[str, Any]) -> Dict[str, Any]:
        data = self.read_table(table_name)
        for item in data:
            if all(item.get(k) == v for k, v in conditions.items()):
                return item
        return None

    def find(
        self, table_name: str, conditions: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        data = self.read_table(table_name)
        if not conditions:
            return data

        results = []
        for item in data:
            if all(item.get(k) == v for k, v in conditions.items()):
                results.append(item)
        return results

    def update(
        self, table_name: str, conditions: Dict[str, Any], new_data: Dict[str, Any]
    ):
        data = self.read_table(table_name)
        updated = False
        for item in data:
            if all(item.get(k) == v for k, v in conditions.items()):
                item.update(new_data)
                updated = True
        if updated:
            self.write_table(table_name, data)

    def delete(self, table_name: str, conditions: Dict[str, Any]):
        data = self.read_table(table_name)
        new_data = [
            item
            for item in data
            if not all(item.get(k) == v for k, v in conditions.items())
        ]
        if len(new_data) < len(data):
            self.write_table(table_name, new_data)

    def generate_id(self, table_name: str) -> int:
        """Generates a unique integer ID for a given table."""
        table_data = self.find(table_name)
        if not table_data:
            return 1
        return max(item.get("id", 0) for item in table_data) + 1


db = JSONDatabase()


def get_db():
    return db
