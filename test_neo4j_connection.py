"""
SCRIPT KIỂM TRA KẾT NỐI NEO4J TỰ ĐỘNG
Dành cho thành viên nhóm kiểm tra kết nối Database trước khi chạy mã nguồn chính.

Cách chạy:
    1. Cài đặt thư viện: pip install neo4j python-dotenv
    2. Điền thông tin vào file .env
    3. Chạy lệnh: python test_neo4j_connection.py
"""

import os
import sys
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from neo4j import GraphDatabase
except ImportError:
    print("❌ Lỗi: Chưa cài đặt thư viện 'neo4j'.")
    print("👉 Hãy chạy lệnh: pip install neo4j python-dotenv")
    sys.exit(1)

def test_connection():
    uri = os.getenv("NEO4J_URI", "neo4j+s://53f3c4e3.databases.neo4j.io")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    print("==================================================")
    print("🔍 ĐANG KIỂM TRA KẾT NỐI CƠ SỞ DỮ LIỆU NEO4J...")
    print("==================================================")
    print(f"📍 URI:      {uri}")
    print(f"👤 Username: {user}")
    print(f"🗄️  Database: {database}")
    
    if not password or password == "<NHAP_MAT_KHAU_VAO_DAY>":
        print("\n⚠️ CẢNH BÁO: Bạn chưa nhập mật khẩu vào biến môi trường NEO4J_PASSWORD!")
        password = input("🔑 Vui lòng nhập mật khẩu Neo4j: ").strip()

    start_time = time.time()
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        latency = (time.time() - start_time) * 1000
        print(f"\n✅ KẾT NỐI THÀNH CÔNG! (Độ trễ xác thực: {latency:.2f} ms)")

        # Truy vấn kiểm tra dữ liệu
        with driver.session(database=database) as session:
            # 1. Đếm tổng số Node
            res_nodes = session.run("MATCH (n) RETURN count(n) AS total_nodes")
            total_nodes = res_nodes.single()["total_nodes"]

            # 2. Đếm số Knowledge Point (KP)
            res_kps = session.run("MATCH (n:KP) RETURN count(n) AS total_kps")
            rec_kp = res_kps.single()
            total_kps = rec_kp["total_kps"] if rec_kp else 0

            # 3. Đếm số quan hệ IS_PREREQUISITE_OF
            res_rels = session.run("MATCH ()-[r:IS_PREREQUISITE_OF]->() RETURN count(r) AS total_prereq")
            rec_rel = res_rels.single()
            total_prereq = rec_rel["total_prereq"] if rec_rel else 0

            print("\n📊 THỐNG KÊ NHANH DỮ LIỆU TRÊN NEO4J:")
            print(f"   • Tổng số Nodes:              {total_nodes:,}")
            print(f"   • Số Điểm kiến thức (:KP):    {total_kps:,}")
            print(f"   • Số Cạnh tiên quyết:         {total_prereq:,}")

        driver.close()
        print("\n🎉 Môi trường Neo4j đã sẵn sàng để phục vụ hệ thống!")

    except Exception as e:
        print(f"\n❌ KẾT NỐI THẤT BẠI!")
        print(f"Chi tiết lỗi: {e}")
        print("\n💡 GỢI Ý KHẮC PHỤC:")
        print("1. Kiểm tra lại mật khẩu xem có bị gõ sai hoặc thừa khoảng trắng không.")
        print("2. Nếu dùng Neo4j Aura Cloud, đảm bảo đường dẫn bắt đầu bằng 'neo4j+s://'.")
        print("3. Kiểm tra xem instance Neo4j Aura có đang bị Pause (tạm dừng) trên console.neo4j.io hay không.")

if __name__ == "__main__":
    test_connection()
