"""Extract and enrich 270 stratified products from H&M Parquet for 300-product DB seed.

Uses real local product photos from C:\\Users\\Administrator\\Downloads\\archive\\images
(matched by category, gender, and color via styles.csv) and copies them to
fashion-shop-be/uploads/products/ so they are served locally at
http://localhost:8080/uploads/products/archive-<id>.jpg.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PARQUET_PATH = ROOT / "data" / "hm_ecommerce_products_enriched.parquet"
OUTPUT_JSON_PATH = ROOT / "data" / "hm_seed_300.json"

ARCHIVE_DIR = Path(r"C:\Users\Administrator\Downloads\archive")
ARCHIVE_IMAGES_DIR = ARCHIVE_DIR / "images"
ARCHIVE_STYLES_CSV = ARCHIVE_DIR / "styles.csv"

BE_UPLOADS_DIR = Path(
    r"H:\Code\PET\e-commerce-clothes\fashion-shop-be\uploads\products"
)
BE_BASE_URL = os.environ.get("MAIN_BE_APP_URL", "http://localhost:8080").rstrip("/")

# Target H&M product counts per subcategory (270 total + 30 anchor = 300 total)
CATEGORY_QUOTAS: dict[str, int] = {
    "Áo thun": 27,
    "Áo polo": 20,
    "Áo sơ mi": 27,
    "Áo khoác": 25,
    "Quần jeans": 22,
    "Quần kaki": 20,
    "Quần short": 18,
    "Quần âu": 21,
    "Váy liền": 25,
    "Chân váy": 20,
    "Đầm dạ hội": 19,
    "Mũ nón": 8,
    "Thắt lưng": 9,
    "Túi xách": 9,
}

COLOR_MAP: dict[str, tuple[str, str]] = {
    "Black": ("Đen", "#1A1A1A"),
    "White": ("Trắng", "#FFFFFF"),
    "Off White": ("Trắng kem", "#FAF9F6"),
    "Dark Blue": ("Xanh navy", "#1B2A4A"),
    "Blue": ("Xanh dương", "#2B5B84"),
    "Light Blue": ("Xanh lam nhạt", "#A4C8E1"),
    "Grey": ("Xám tiêu", "#8A8D8F"),
    "Dark Grey": ("Xám đậm", "#4A4D50"),
    "Light Grey": ("Xám nhạt", "#D1D3D4"),
    "Beige": ("Be sữa", "#F3EDE2"),
    "Light Beige": ("Be sáng", "#F7F2E7"),
    "Dark Beige": ("Nâu be", "#C8B59B"),
    "Greenish Khaki": ("Xanh rêu khaki", "#5E6752"),
    "Dark Green": ("Xanh cổ vịt", "#1E4D40"),
    "Green": ("Xanh lá", "#2E7D32"),
    "Light Green": ("Xanh mint", "#A8E6CF"),
    "Red": ("Đỏ tươi", "#C62828"),
    "Dark Red": ("Đỏ đô", "#7A1C24"),
    "Pink": ("Hồng", "#E91E63"),
    "Light Pink": ("Hồng pastel", "#F4C2C2"),
    "Dark Pink": ("Hồng đất", "#AD5D5D"),
    "Yellow": ("Vàng mù tạt", "#D4AF37"),
    "Light Yellow": ("Vàng kem", "#FFF59D"),
    "Yellowish Brown": ("Nâu camel", "#C19A6B"),
    "Light Orange": ("Cam đào", "#FFAB91"),
    "Dark Orange": ("Cam đất", "#BF5B30"),
    "Orange": ("Cam", "#EF6C00"),
    "Gold": ("Vàng ánh kim", "#D4AF37"),
    "Silver": ("Bạc", "#C0C0C0"),
    "Light Turquoise": ("Xanh ngọc", "#80CBC4"),
    "Turquoise": ("Xanh ngọc bích", "#26A69A"),
    "Light Purple": ("Tím lilac", "#CE93D8"),
    "Dark Purple": ("Tím than", "#311B92"),
}

ARCHIVE_COLOR_TO_VI: dict[str, list[str]] = {
    "Black": ["Đen", "Xám đậm"],
    "White": ["Trắng", "Trắng kem"],
    "Off White": ["Trắng kem", "Trắng", "Be sáng"],
    "Cream": ["Be sữa", "Be sáng", "Trắng kem"],
    "Beige": ["Be sữa", "Be sáng", "Nâu be"],
    "Navy Blue": ["Xanh navy", "Tím than"],
    "Blue": ["Xanh dương", "Xanh lam nhạt", "Xanh navy"],
    "Teal": ["Xanh cổ vịt", "Xanh ngọc", "Xanh ngọc bích"],
    "Turquoise Blue": ["Xanh ngọc", "Xanh ngọc bích", "Xanh lam nhạt"],
    "Grey": ["Xám tiêu", "Xám nhạt", "Xám đậm", "Bạc"],
    "Charcoal": ["Xám đậm", "Đen", "Xám tiêu"],
    "Grey Melange": ["Xám tiêu", "Xám nhạt"],
    "Silver": ["Bạc", "Xám nhạt"],
    "Green": ["Xanh lá", "Xanh cổ vịt", "Xanh mint"],
    "Olive": ["Xanh rêu khaki", "Xanh lá"],
    "Khaki": ["Xanh rêu khaki", "Nâu be", "Be sữa"],
    "Red": ["Đỏ tươi", "Đỏ đô"],
    "Maroon": ["Đỏ đô", "Hồng đất"],
    "Pink": ["Hồng", "Hồng pastel", "Hồng đất"],
    "Peach": ["Hồng pastel", "Cam đào"],
    "Magenta": ["Hồng", "Đỏ đô"],
    "Purple": ["Tím than", "Tím lilac"],
    "Lavender": ["Tím lilac", "Hồng pastel"],
    "Yellow": ["Vàng mù tạt", "Vàng kem", "Vàng ánh kim"],
    "Gold": ["Vàng ánh kim", "Vàng mù tạt"],
    "Brown": ["Nâu camel", "Nâu be", "Cam đất"],
    "Tan": ["Nâu camel", "Nâu be"],
    "Orange": ["Cam", "Cam đào", "Cam đất"],
    "Rust": ["Cam đất", "Đỏ đô"],
}

FALLBACK_COLORS: list[tuple[str, str]] = [
    ("Đen", "#1A1A1A"),
    ("Trắng", "#FFFFFF"),
    ("Xanh navy", "#1B2A4A"),
    ("Be sữa", "#F3EDE2"),
    ("Xám tiêu", "#8A8D8F"),
]

PATTERN_VI: dict[str, str] = {
    "Solid": "Trơn Basic",
    "All over pattern": "Họa Tiết Toàn Thân",
    "Melange": "Xước Melange",
    "Stripe": "Kẻ Sọc",
    "Denim": "Denim Wash",
    "Front print": "In Hình Trước Ngực",
    "Placement print": "In Họa Tiết Điểm Nhấn",
    "Check": "Kẻ Caro",
    "Colour blocking": "Phối Màu Colorblock",
    "Lace": "Phối Ren",
    "Embroidery": "Thêu Tinh Tế",
    "Glittering/Metallic": "Ánh Nhũ Sang Trọng",
    "Jacquard": "Dệt Hoa Văn Jacquard",
    "Rib": "Dệt Gân Tăm",
    "Mesh": "Phối Lưới Thoáng Khí",
}

# Rich Factorial Profiles per Category
CATEGORY_PROFILES: dict[str, list[dict[str, str]]] = {
    "Áo thun": [
        {
            "style": "Regular Fit Cổ Tròn",
            "material": "100% Cotton Compact 250gsm",
            "season": "Xuân Hè",
            "care": "Giặt máy nhẹ nước lạnh, không dùng chất tẩy",
            "occasion": (
                "mặc hàng ngày, đi chơi, dạo phố cuối tuần hoặc mặc lót blazer công sở"
            ),
            "body_fit": (
                "Form Regular Fit đứng phom, chất vải dày dặn giúp người dáng "
                "gầy cao trông đầy đặn hơn và người có bụng che khuyết điểm tốt"
            ),
        },
        {
            "style": "Oversize Streetwear",
            "material": "Cotton 2 chiều 240gsm thoáng mát",
            "season": "Bốn mùa",
            "care": "Lộn trái khi giặt và phơi, không ủi trực tiếp lên bề mặt in",
            "occasion": (
                "dạo phố, đi cà phê, tụ tập bạn bè phong cách trẻ trung cá tính"
            ),
            "body_fit": (
                "Form rộng Oversize giấu dáng thoải mái, phù hợp cả bạn gầy muốn "
                "ăn gian cân nặng lẫn bạn đậm người thích sự phóng khoáng"
            ),
        },
        {
            "style": "Slim Fit Co Giãn 4 Chiều",
            "material": "95% Cotton USA, 5% Spandex",
            "season": "Xuân Hè",
            "care": "Giặt máy chế độ nhẹ, phơi nơi thoáng mát",
            "occasion": "đi chơi, hẹn hò hoặc vận động nhẹ",
            "body_fit": (
                "Form Slim Fit ôm gọn tôn dáng cơ thể săn chắc, co giãn đàn hồi tốt"
            ),
        },
        {
            "style": "Thể Thao Thoáng Khí Quick-Dry",
            "material": "Polyester Coolmax thấm hút mồ hôi",
            "season": "Mùa hè",
            "care": "Giặt nhanh, mau khô, không cần ủi",
            "occasion": "đi biển, du lịch mùa hè, chạy bộ và vận động ngoài trời",
            "body_fit": (
                "Trọng lượng siêu nhẹ, bề mặt vải mát lạnh, "
                "thoát ẩm nhanh cho ngày nắng nóng"
            ),
        },
    ],
    "Áo polo": [
        {
            "style": "Pique Công Sở Thanh Lịch",
            "material": "95% Cotton Pique mắt chim, 5% Spandex",
            "season": "Bốn mùa",
            "care": "Giặt nước lạnh, ủi nhiệt độ trung bình, cài cúc cổ khi phơi",
            "occasion": (
                "đi làm công sở lịch sự, gặp gỡ đối tác hoặc đi cà phê cuối tuần"
            ),
            "body_fit": (
                "Form Regular Fit chuẩn mực, cổ bẻ đứng dáng "
                "giúp che bụng tốt và tôn vai"
            ),
        },
        {
            "style": "Thể Thao Co Giãn Thoáng Mát",
            "material": "Vải Café Coolmax khử mùi, chống nhăn",
            "season": "Mùa hè",
            "care": "Giặt máy bình thường, không cần ủi thường xuyên",
            "occasion": "đi làm, chơi golf, du lịch biển hoặc vận động ngày hè",
            "body_fit": (
                "Chất vải mát lạnh thấm hút mồ hôi vượt trội, "
                "co giãn 4 chiều thoải mái suốt ngày dài"
            ),
        },
        {
            "style": "Phối Bo Cổ Smart-Casual",
            "material": "100% Cotton Compact dệt kim",
            "season": "Xuân Hè",
            "care": "Giặt nhẹ, phơi ngang hoặc dùng móc vai bản rộng",
            "occasion": "đi hẹn hò, dạo phố, đi làm phong cách trẻ trung",
            "body_fit": (
                "Dáng suông vừa vặn (Regular), dễ phối cùng quần âu, "
                "quần kaki hoặc quần short"
            ),
        },
    ],
    "Áo sơ mi": [
        {
            "style": "Oxford Công Sở Regular Fit",
            "material": "100% Cotton Oxford cao cấp đứng phom",
            "season": "Bốn mùa",
            "care": "Giặt máy nhẹ, ủi ở nhiệt độ trung bình",
            "occasion": "đi làm văn phòng công sở, họp hành, sự kiện lịch sự",
            "body_fit": (
                "Form Regular Fit suông nhẹ giúp người đậm người che bụng bia "
                "hiệu quả và người gầy trông cân đối hơn"
            ),
        },
        {
            "style": "Bamboo Chống Nhăn Slim Fit",
            "material": "70% Sợi tre Bamboo chống nhăn, 30% Microfiber",
            "season": "Bốn mùa",
            "care": (
                "Vải chống nhăn tự nhiên (Non-iron), giặt máy nhẹ, "
                "treo thẳng không cần ủi thường xuyên"
            ),
            "occasion": "đi làm công sở bận rộn, dự tiệc cưới, gặp khách hàng",
            "body_fit": (
                "Form Slim Fit ôm vừa vặn tôn dáng, "
                "bề mặt vải mềm mượt mát tay và kháng khuẩn"
            ),
        },
        {
            "style": "Linen Thoáng Mát Dáng Thư Giãn",
            "material": "100% Linen tự nhiên thoáng khí",
            "season": "Mùa hè",
            "care": "Giặt nước mát, không vắt xoắn mạnh, ủi hơi nước khi còn ẩm",
            "occasion": "đi du lịch biển, dạo phố mùa hè, nghỉ dưỡng resort",
            "body_fit": (
                "Form Relaxed Fit rộng thoáng, "
                "thấm hút mồ hôi cực tốt cho thời tiết nắng nóng"
            ),
        },
        {
            "style": "Lụa Mềm Thanh Lịch",
            "material": "Lụa Satin pha Cotton mềm mịn",
            "season": "Xuân Hè",
            "care": "Giặt tay hoặc bỏ túi giặt, ủi hơi nước nhiệt độ thấp",
            "occasion": "đi làm công sở, dự tiệc nhẹ, hẹn hò thanh lịch",
            "body_fit": (
                "Độ rủ nhẹ tự nhiên tôn nét mềm mại, thoáng mát và sang trọng"
            ),
        },
    ],
    "Áo khoác": [
        {
            "style": "Blazer Công Sở Hàn Quốc",
            "material": "Kaki Tuyết Mưa cao cấp có lớp lót lụa",
            "season": "Bốn mùa",
            "care": "Khuyến khích giặt khô hoặc giặt hấp để giữ phom vai",
            "occasion": "đi làm công sở, dự tiệc cưới, hội nghị thanh lịch",
            "body_fit": (
                "Cầu vai đứng phom giúp tôn dáng, che khuyết điểm vòng hai "
                "và dễ phối cùng áo thun hoặc sơ mi"
            ),
        },
        {
            "style": "Gió Windbreaker Tráng PU Chống Nước",
            "material": "Polyester 2 lớp tráng PU cản gió chống thấm nhẹ",
            "season": "Thu Đông",
            "care": "Giặt tay nhẹ, không vắt mạnh, phơi trong bóng râm",
            "occasion": (
                "đi phượt, du lịch, khoác ngoài chống nắng hoặc giữ ấm ngày se lạnh"
            ),
            "body_fit": (
                "Form Regular năng động, có khóa kéo và mũ tiện dụng cho mọi vóc dáng"
            ),
        },
        {
            "style": "Hoodie & Sweater Nỉ Bông Giữ Ấm",
            "material": "Nỉ bông Cotton 320gsm dày dặn mềm mịn",
            "season": "Mùa đông",
            "care": "Lộn mặt trái khi giặt, tránh sấy nhiệt độ cao",
            "occasion": "mặc mùa đông lạnh, đi học, đi chơi dạo phố",
            "body_fit": "Form rộng rãi ấm áp, tạo cảm giác đầy đặn trẻ trung",
        },
        {
            "style": "Bomber & Cardigan Thời Trang",
            "material": "Dù Nylon lót chần bông / Len dệt kim mềm",
            "season": "Thu Đông",
            "care": "Giặt nhẹ bằng nước lạnh, phơi phẳng",
            "occasion": "dạo phố thu đông, đi làm, hẹn hò",
            "body_fit": ("Chiều dài vừa ngang hông giúp hack chiều cao và gọn người"),
        },
    ],
    "Quần jeans": [
        {
            "style": "Slim Fit Co Giãn Tôn Dáng",
            "material": "98% Cotton Denim 12oz, 2% Elastane co giãn",
            "season": "Bốn mùa",
            "care": "Giặt mặt trái với nước lạnh, hạn chế sấy nóng",
            "occasion": "đi làm smart-casual, đi chơi, dạo phố hàng ngày",
            "body_fit": (
                "Ống ôm vừa vặn Slim Fit giúp đôi chân trông thon dài "
                "mà vẫn đứng ngồi thoải mái"
            ),
        },
        {
            "style": "Cạp Cao Ống Rộng Hack Chiều Cao",
            "material": "100% Cotton Denim đứng phom",
            "season": "Bốn mùa",
            "care": "Giặt riêng trong 2 lần đầu, phơi mặt trái",
            "occasion": ("đi chơi, dạo phố, chụp ảnh phong cách Retro trẻ trung"),
            "body_fit": (
                "Thiết kế cạp cao trên rốn kết hợp ống suông rộng giúp hack "
                "chiều cao tối đa cho người nấm lùn và che khuyết điểm bắp chân"
            ),
        },
        {
            "style": "Straight Fit Ống Đứng Cổ Điển",
            "material": "Denim Cotton pha Spandex bền màu",
            "season": "Bốn mùa",
            "care": "Giặt máy bình thường, không dùng thuốc tẩy",
            "occasion": "đi học, đi làm, đi du lịch bụi",
            "body_fit": (
                "Form ống đứng Straight Fit phù hợp cả người chân gầy "
                "lẫn người có đùi to"
            ),
        },
    ],
    "Quần kaki": [
        {
            "style": "Chino Ống Đứng Công Sở",
            "material": "97% Cotton Twill Kaki, 3% Spandex co giãn nhẹ",
            "season": "Bốn mùa",
            "care": "Giặt máy bình thường, ủi mặt trái nhiệt độ vừa",
            "occasion": ("đi làm công sở lịch sự, đi gặp khách hàng hoặc đi chơi"),
            "body_fit": (
                "Form Regular ống đứng lịch lãm, chất kaki dày dặn giúp tôn dáng "
                "và dễ phối cùng áo sơ mi hoặc áo polo"
            ),
        },
        {
            "style": "Jogger & Relaxed Năng Động",
            "material": "Kaki Cotton chun co giãn 4 chiều",
            "season": "Bốn mùa",
            "care": "Giặt máy chế độ thường, nhanh khô",
            "occasion": "đi dạo phố, du lịch, di chuyển đường dài thoải mái",
            "body_fit": (
                "Lưng thun co giãn dễ chịu cho người có bụng, form gọn gàng trẻ trung"
            ),
        },
        {
            "style": "Cạp Cao Ống Suông Thanh Lịch",
            "material": "Kaki mềm cao cấp ít nhăn",
            "season": "Xuân Hè",
            "care": "Giặt nhẹ, treo thẳng khi phơi",
            "occasion": "đi làm văn phòng, đi cà phê cuối tuần",
            "body_fit": (
                "Thiết kế cạp cao giúp kéo dài tỷ lệ chân, hack chiều cao hiệu quả"
            ),
        },
    ],
    "Quần short": [
        {
            "style": "Chino Dạo Phố Trên Gối",
            "material": "Cotton Chino thoáng mát thấm hút tốt",
            "season": "Mùa hè",
            "care": "Giặt máy bình thường, ủi nhẹ",
            "occasion": "đi biển, đi dạo phố mùa hè, du lịch nghỉ dưỡng",
            "body_fit": (
                "Độ dài trên gối vừa vặn giúp hack chiều cao, tạo vẻ ngoài năng động"
            ),
        },
        {
            "style": "Thể Thao Quick-Dry Thoáng Khí",
            "material": "Polyester pha Spandex siêu nhẹ mau khô",
            "season": "Mùa hè",
            "care": "Giặt máy thoải mái, khô nhanh trong 30 phút",
            "occasion": ("đi biển, tập thể thao, chạy bộ, mặc ở nhà ngày nắng nóng"),
            "body_fit": "Cạp chun co giãn linh hoạt, ống quần rộng thoáng tối đa",
        },
        {
            "style": "Cạp Cao Tôn Dáng Trẻ Trung",
            "material": "Kaki / Denim mềm đứng phom",
            "season": "Xuân Hè",
            "care": "Giặt nước lạnh, phơi nơi thoáng gió",
            "occasion": "đi chơi cuối tuần, du lịch biển, dạo phố",
            "body_fit": "Cạp cao ôm gọn vòng eo, tôn dáng chân dài cho bạn nữ",
        },
    ],
    "Quần âu": [
        {
            "style": "Tây Âu Xếp Ly Công Sở",
            "material": "Wool Blend pha Rayon chống nhăn cao cấp",
            "season": "Bốn mùa",
            "care": "Giặt tay nhẹ hoặc giặt khô, ủi ly bằng bàn ủi hơi nước",
            "occasion": "đi làm công sở, dự tiệc cưới, hội họp trang trọng",
            "body_fit": (
                "Form Straight / Regular xếp ly trước giúp người có bụng mặc "
                "thoải mái và giữ phom đứng sang trọng"
            ),
        },
        {
            "style": "Cạp Cao Ống Suông Hack Dáng",
            "material": "Vải Tuyết Mưa cao cấp dày dặn không nhăn",
            "season": "Bốn mùa",
            "care": "Giặt máy túi giặt, treo thẳng không cần ủi nhiều",
            "occasion": "đi làm văn phòng, gặp gỡ đối tác, dự sự kiện",
            "body_fit": (
                "Cạp cao kết hợp ống suông thẳng từ hông xuống mắt cá giúp hack "
                "chiều cao cho người nấm lùn và che khuyết điểm chân"
            ),
        },
        {
            "style": "Slim Fit Tailored Lịch Lãm",
            "material": "Poly-Viscose co giãn nhẹ đứng phom",
            "season": "Bốn mùa",
            "care": "Giặt nhẹ, ủi nhiệt độ trung bình",
            "occasion": "đi làm, mặc cùng áo sơ mi hoặc áo blazer",
            "body_fit": ("Ôm gọn gàng tôn chiều cao, tạo phong thái chỉn chu hiện đại"),
        },
    ],
    "Váy liền": [
        {
            "style": "Đầm Công Sở Chiết Eo Thanh Lịch",
            "material": "Cotton Lụa cao cấp có lớp lót mềm",
            "season": "Xuân Hè",
            "care": "Giặt tay hoặc giặt máy chế độ nhẹ, ủi hơi nước",
            "occasion": "đi làm công sở, gặp khách hàng, đi họp lịch sự",
            "body_fit": (
                "Thiết kế chiết eo cao kèm tùng váy chữ A xòe nhẹ giúp tôn "
                "vòng eo và giấu bụng dưới khéo léo"
            ),
        },
        {
            "style": "Váy Suông Dạo Phố Thoáng Mát",
            "material": "Voan Lụa / Linen mềm mát thấm hút tốt",
            "season": "Mùa hè",
            "care": "Giặt tay nhẹ nhàng với nước mát, phơi bóng râm",
            "occasion": "đi chơi cuối tuần, đi du lịch biển, dạo phố cà phê",
            "body_fit": (
                "Dáng suông thoải mái che khuyết điểm vòng hai và hông, "
                "phù hợp nhiều vóc dáng"
            ),
        },
        {
            "style": "Váy Dệt Kim Tôn Dáng Thu Đông",
            "material": "Len Acrylic dệt kim co giãn mềm mịn",
            "season": "Thu Đông",
            "care": "Phơi nằm ngang tránh dão vải, giặt nước lạnh",
            "occasion": "đi hẹn hò, dạo phố mùa thu đông, dự tiệc tối",
            "body_fit": (
                "Ôm vừa vặn tôn đường cong cơ thể, giữ ấm tốt mà vẫn thanh thoát"
            ),
        },
    ],
    "Chân váy": [
        {
            "style": "Chữ A Cạp Cao Xếp Ly",
            "material": "Vải Tuyết Mưa dày dặn có quần bảo hộ bên trong",
            "season": "Bốn mùa",
            "care": "Giặt nhẹ, treo thẳng khi phơi để giữ nếp ly",
            "occasion": "đi học, đi chơi dạo phố, đi làm trẻ trung",
            "body_fit": (
                "Cạp cao ôm eo kết hợp dáng chữ A giúp hack chiều cao tối đa "
                "cho bạn nữ nấm lùn"
            ),
        },
        {
            "style": "Midi Công Sở Xẻ Tà Thanh Lịch",
            "material": "Kaki Lụa / Denim co giãn nhẹ đứng phom",
            "season": "Bốn mùa",
            "care": "Giặt máy chế độ nhẹ, ủi nhiệt độ vừa",
            "occasion": ("đi làm văn phòng công sở, đi cà phê, dự sự kiện nhẹ"),
            "body_fit": (
                "Chiều dài Midi qua gối che khuyết điểm bắp chân và tạo nét duyên dáng"
            ),
        },
    ],
    "Đầm dạ hội": [
        {
            "style": "Dự Tiệc Phối Ren Thêu Hoa Sang Trọng",
            "material": "Ren thêu cao cấp lót lụa Satin mềm mại",
            "season": "Bốn mùa",
            "care": "Giặt khô chuyên dụng hoặc giặt tay cực nhẹ",
            "occasion": (
                "đi tiệc cưới khách sạn, dạ hội, sự kiện quan trọng, gala dinner"
            ),
            "body_fit": (
                "Thiết kế ôm ngực chiết eo cao và tùng váy xòe bồng bềnh "
                "giúp tôn dáng kiêu sa và che bụng tốt"
            ),
        },
        {
            "style": "Maxi Lụa Satin Dự Tiệc Quyến Rũ",
            "material": "Lụa Satin bóng mờ cao cấp rủ mềm",
            "season": "Bốn mùa",
            "care": "Giặt tay nước lạnh, ủi hơi nước mặt trái",
            "occasion": ("đi đám cưới bạn thân, tiệc tối sang trọng, sự kiện lễ hội"),
            "body_fit": (
                "Dáng Maxi thướt tha giúp kéo dài vóc dáng, tôn nét thanh lịch quý phái"
            ),
        },
    ],
    "Mũ nón": [
        {
            "style": "Bucket & Lưỡi Trai Dạo Phố",
            "material": "100% Cotton Canvas / Dù chống nắng thoáng khí",
            "season": "Bốn mùa",
            "care": "Giặt tay bằng bàn chải mềm, phơi nơi thoáng gió",
            "occasion": "đi biển, du lịch, dạo phố, hoạt động ngoài trời",
            "body_fit": (
                "Freesize có khóa điều chỉnh vòng đầu linh hoạt, che nắng tốt"
            ),
        }
    ],
    "Thắt lưng": [
        {
            "style": "Da Bản Chuẩn Khóa Kim Loại",
            "material": "Da thật cao cấp bền đẹp, mặt khóa hợp kim chống gỉ",
            "season": "Bốn mùa",
            "care": "Lau khô bằng khăn mềm, bảo quản nơi khô ráo",
            "occasion": (
                "phối cùng quần âu công sở, quần jeans, quần kaki hoặc nhấn eo váy đầm"
            ),
            "body_fit": ("Freesize dễ dàng điều chỉnh nấc vừa vặn với mọi vòng eo"),
        }
    ],
    "Túi xách": [
        {
            "style": "Tote & Đeo Chéo Thời Trang",
            "material": "Da PU cao cấp / Canvas dày dặn 12oz chống thấm nhẹ",
            "season": "Bốn mùa",
            "care": "Lau sạch bề mặt bằng khăn ẩm, tránh chất tẩy mạnh",
            "occasion": "đi làm công sở, đi học, đi dạo phố hoặc du lịch",
            "body_fit": (
                "Ngăn chứa rộng rãi tiện dụng, dễ phối với nhiều phong cách trang phục"
            ),
        }
    ],
}

BRANDS = ["H&M", "Uniqlo", "Zara", "Nike", "Adidas"]

# Price tiers (list_price in VND)
PRICE_TIERS: list[tuple[int, int]] = [
    (179000, 289000),  # Tier 1: Budget (< 300k)
    (329000, 529000),  # Tier 2: Mid-range (300k - 550k)
    (599000, 1150000),  # Tier 3: Premium (600k+)
]


class LocalArchiveImageMatcher:
    """Matches (category, gender, color) to real .jpg files in Downloads\\archive."""

    def __init__(self) -> None:
        BE_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        styles_df = pd.read_csv(ARCHIVE_STYLES_CSV, on_bad_lines="skip")
        existing_files = set(os.listdir(ARCHIVE_IMAGES_DIR))
        styles_df["img_file"] = styles_df["id"].astype(str) + ".jpg"
        styles_df = styles_df[
            styles_df["img_file"].isin(existing_files)
            & styles_df["gender"].isin(["Men", "Women", "Unisex"])
        ].copy()

        styles_df["cat"] = styles_df.apply(self._map_archive_cat, axis=1)
        styles_df = styles_df.dropna(subset=["cat"])

        # Index pools: (cat, gender, vi_color), (cat, vi_color), (cat, gender), (cat)
        self.by_cat_gender_color: dict[tuple[str, str, str], list[int]] = defaultdict(
            list
        )
        self.by_cat_color: dict[tuple[str, str], list[int]] = defaultdict(list)
        self.by_cat_gender: dict[tuple[str, str], list[int]] = defaultdict(list)
        self.by_cat: dict[str, list[int]] = defaultdict(list)
        self.cursors: dict[Any, int] = defaultdict(int)
        self.copied_ids: set[int] = set()

        gender_map = {"Men": "MALE", "Women": "FEMALE", "Unisex": "UNISEX"}

        for _, row in styles_df.iterrows():
            aid = int(row["id"])
            cat = str(row["cat"])
            g = gender_map.get(str(row["gender"]), "UNISEX")
            raw_col = str(row["baseColour"])
            vi_colors = ARCHIVE_COLOR_TO_VI.get(raw_col, [])

            self.by_cat[cat].append(aid)
            self.by_cat_gender[(cat, g)].append(aid)
            if g == "UNISEX":
                self.by_cat_gender[(cat, "MALE")].append(aid)
                self.by_cat_gender[(cat, "FEMALE")].append(aid)

            for vc in vi_colors:
                self.by_cat_color[(cat, vc)].append(aid)
                self.by_cat_gender_color[(cat, g, vc)].append(aid)
                if g in ("MALE", "FEMALE"):
                    self.by_cat_gender_color[(cat, "UNISEX", vc)].append(aid)

    @staticmethod
    def _map_archive_cat(row: pd.Series) -> str | None:
        at = str(row["articleType"])
        usage = str(row["usage"])
        name = str(row["productDisplayName"]).lower()
        if at in ["Caps", "Hat"]:
            return "Mũ nón"
        if at == "Belts":
            return "Thắt lưng"
        if at in [
            "Handbags",
            "Backpacks",
            "Clutches",
            "Duffel Bag",
            "Messenger Bag",
            "Mobile Pouch",
            "Laptop Bag",
        ]:
            return "Túi xách"
        if at == "Skirts":
            return "Chân váy"
        if at == "Dresses":
            if usage in ["Party", "Formal", "Ethnic"] or any(
                k in name
                for k in [
                    "party",
                    "maxi",
                    "lace",
                    "satin",
                    "evening",
                    "gown",
                    "bodycon",
                ]
            ):
                return "Đầm dạ hội"
            return "Váy liền"
        if at == "Shorts":
            return "Quần short"
        if at == "Jeans":
            return "Quần jeans"
        if at in ["Trousers", "Track Pants", "Capris", "Chinos"]:
            if usage == "Formal" or "formal" in name or "trouser" in name:
                return "Quần âu"
            return "Quần kaki"
        if at == "Tshirts" and ("polo" in name or "collar" in name):
            return "Áo polo"
        if at in ["Shirts", "Tops", "Tunics"]:
            if at == "Shirts" or "shirt" in name or "blouse" in name:
                return "Áo sơ mi"
            return "Áo thun"
        if at in [
            "Jackets",
            "Sweatshirts",
            "Sweaters",
            "Blazers",
            "Shrug",
            "Nehru Jackets",
            "Rain Jacket",
            "Waistcoat",
        ]:
            return "Áo khoác"
        if at == "Tshirts":
            return "Áo thun"
        return None

    def pick_and_copy_image(
        self, category_name: str, target_gender: str, vi_color: str
    ) -> str:
        """Select a matching local .jpg, copy to BE uploads/products, return URL."""
        key1 = (category_name, target_gender, vi_color)
        key2 = (category_name, vi_color)
        key3 = (category_name, target_gender)
        key4 = category_name

        if self.by_cat_gender_color[key1]:
            pool = self.by_cat_gender_color[key1]
            ckey: Any = ("cgc", key1)
        elif self.by_cat_color[key2]:
            pool = self.by_cat_color[key2]
            ckey = ("cc", key2)
        elif self.by_cat_gender[key3]:
            pool = self.by_cat_gender[key3]
            ckey = ("cg", key3)
        else:
            pool = self.by_cat[key4]
            ckey = ("c", key4)

        idx = self.cursors[ckey] % len(pool)
        self.cursors[ckey] += 1
        archive_id = pool[idx]

        filename = f"archive-{archive_id}.jpg"
        dest_path = BE_UPLOADS_DIR / filename
        if archive_id not in self.copied_ids:
            src_path = ARCHIVE_IMAGES_DIR / f"{archive_id}.jpg"
            if not dest_path.exists():
                shutil.copy2(src_path, dest_path)
            self.copied_ids.add(archive_id)

        return f"{BE_BASE_URL}/uploads/products/{filename}"


def map_category(row: pd.Series) -> str | None:
    pt = str(row["product_type_name"])
    gg = str(row["garment_group_name"])
    ga = str(row["graphical_appearance_name"])
    name = str(row["prod_name"]).lower()
    desc = str(row["detail_desc"]).lower()

    if pt in ["Hat/beanie", "Cap/peaked", "Hat/brim"]:
        return "Mũ nón"
    if pt == "Belt":
        return "Thắt lưng"
    if pt == "Bag":
        return "Túi xách"
    if pt == "Skirt":
        return "Chân váy"
    if pt == "Dress":
        if ga in ["Lace", "Embroidery", "Glittering/Metallic"] or any(
            k in name or k in desc
            for k in [
                "lace",
                "satin",
                "sequin",
                "party",
                "maxi",
                "evening",
                "chiffon",
            ]
        ):
            return "Đầm dạ hội"
        return "Váy liền"
    if pt == "Shorts":
        return "Quần short"
    if pt == "Trousers":
        if (
            gg == "Trousers Denim"
            or ga == "Denim"
            or "jeans" in name
            or "denim" in desc
        ):
            return "Quần jeans"
        if gg == "Dressed" or any(
            k in name or k in desc
            for k in ["suit", "tailored", "crease", "slacks", "formal"]
        ):
            return "Quần âu"
        return "Quần kaki"
    if any(k in name or k in desc for k in ["polo", "piqué", "pique"]) and pt in [
        "T-shirt",
        "Shirt",
        "Top",
        "Sweater",
    ]:
        return "Áo polo"
    if pt in ["Shirt", "Blouse"]:
        return "Áo sơ mi"
    if pt in ["Jacket", "Blazer", "Hoodie", "Coat", "Cardigan", "Sweater"]:
        return "Áo khoác"
    if pt in ["T-shirt", "Top", "Vest top"]:
        return "Áo thun"
    return None


def map_gender(index_group_name: str, category_name: str, idx: int) -> str:
    if category_name in ["Váy liền", "Chân váy", "Đầm dạ hội"]:
        return "FEMALE"
    if index_group_name == "Menswear":
        return "MALE"
    if index_group_name == "Ladieswear":
        if category_name in ["Mũ nón", "Túi xách"] and idx % 3 == 0:
            return "UNISEX"
        return "FEMALE"
    mod = idx % 3
    if mod == 0:
        return "UNISEX"
    if mod == 1:
        return "MALE"
    return "FEMALE"


def get_sizes_for_product(category_name: str, target_gender: str) -> list[str]:
    if category_name in ["Mũ nón", "Thắt lưng", "Túi xách"]:
        return ["Freesize"]
    if (
        category_name in ["Quần jeans", "Quần kaki", "Quần âu"]
        and target_gender == "MALE"
    ):
        return ["29", "30", "31", "32", "34"]
    if target_gender == "FEMALE":
        return ["S", "M", "L", "XL"]
    return ["S", "M", "L", "XL", "XXL"]


def build_vietnamese_name(
    category_name: str,
    target_gender: str,
    profile_style: str,
    graphical_appearance: str,
    raw_prod_name: str,
    product_code: str,
) -> str:
    gender_label = {
        "MALE": "Nam",
        "FEMALE": "Nữ",
        "UNISEX": "Unisex",
    }[target_gender]

    pattern_vi = PATTERN_VI.get(graphical_appearance, "Thời Trang")
    clean_tag = "".join(c for c in raw_prod_name if c.isalnum() or c == " ").strip()
    first_word = clean_tag.split()[0].capitalize() if clean_tag else "Classic"
    if first_word.lower() in [
        "dress",
        "shirt",
        "trousers",
        "shorts",
        "skirt",
        "top",
        "jacket",
        "sweater",
        "hoodie",
        "bag",
        "belt",
        "hat",
        "cap",
    ]:
        first_word = f"HM{product_code[-3:]}"

    return f"{category_name} {gender_label} {profile_style} {pattern_vi} {first_word}"


def main() -> None:
    rng = random.Random(20261121)
    matcher = LocalArchiveImageMatcher()
    df = pd.read_parquet(PARQUET_PATH)

    # Filter adult groups with valid descriptions and image URLs
    adult = df[
        df["index_group_name"].isin(["Ladieswear", "Menswear", "Divided", "Sport"])
        & df["image_url"].notna()
        & df["detail_desc"].notna()
    ].copy()

    adult["mapped_cat"] = adult.apply(map_category, axis=1)
    adult = adult.dropna(subset=["mapped_cat"])

    # Group by (mapped_cat, product_code) to prefer multiple color articles
    grouped_meta = (
        adult.groupby(["mapped_cat", "product_code"])
        .agg(
            article_count=("article_id", "count"),
            index_group=("index_group_name", "first"),
        )
        .reset_index()
    )

    selected_products: list[dict[str, Any]] = []
    global_seen_codes: set[str] = set()
    variant_seq = 1

    for cat_name, quota in CATEGORY_QUOTAS.items():
        cat_candidates = grouped_meta[
            (grouped_meta["mapped_cat"] == cat_name)
            & (~grouped_meta["product_code"].isin(global_seen_codes))
        ].copy()
        cat_candidates["priority"] = cat_candidates["article_count"].clip(upper=4)
        cat_candidates = cat_candidates.sort_values(
            by=["priority", "product_code"], ascending=[False, True]
        )

        # Stratified selection across Menswear / Ladieswear / Divided+Sport
        if cat_name not in ["Váy liền", "Chân váy", "Đầm dạ hội"]:
            men_pool = cat_candidates[cat_candidates["index_group"] == "Menswear"]
            lady_pool = cat_candidates[cat_candidates["index_group"] == "Ladieswear"]
            other_pool = cat_candidates[
                ~cat_candidates["index_group"].isin(["Menswear", "Ladieswear"])
            ]

            men_target = quota // 3 + (1 if quota % 3 > 0 else 0)
            lady_target = quota // 3 + (1 if quota % 3 > 1 else 0)
            other_target = quota - men_target - lady_target

            picked_codes = list(men_pool["product_code"].head(men_target))
            picked_codes += list(lady_pool["product_code"].head(lady_target))
            picked_codes += list(other_pool["product_code"].head(other_target))

            if len(picked_codes) < quota:
                remaining = [
                    c
                    for c in cat_candidates["product_code"]
                    if c not in set(picked_codes)
                ]
                picked_codes += remaining[: quota - len(picked_codes)]
        else:
            picked_codes = list(cat_candidates["product_code"].head(quota))

        global_seen_codes.update(picked_codes)
        profiles = CATEGORY_PROFILES[cat_name]

        for idx, p_code in enumerate(picked_codes):
            rows = adult[
                (adult["mapped_cat"] == cat_name) & (adult["product_code"] == p_code)
            ].head(3)
            first_row = rows.iloc[0]

            target_gender = map_gender(
                str(first_row["index_group_name"]), cat_name, idx
            )
            profile = profiles[idx % len(profiles)]
            tier_idx = idx % len(PRICE_TIERS)
            low_p, high_p = PRICE_TIERS[tier_idx]
            list_price = (rng.randint(low_p // 1000, high_p // 1000)) * 1000
            discount_ratio = [0.0, 0.0, 0.1, 0.15, 0.2][idx % 5]
            selling_price = int(
                round((list_price * (1.0 - discount_ratio)) / 1000) * 1000
            )

            ga = str(first_row["graphical_appearance_name"])
            raw_name = str(first_row["prod_name"])
            vi_name = build_vietnamese_name(
                cat_name,
                target_gender,
                profile["style"],
                ga,
                raw_name,
                str(p_code),
            )
            digest = hashlib.md5(f"{cat_name}-{p_code}".encode()).hexdigest()[:6]
            slug = f"hm-{p_code}-{digest}"

            # Extract distinct colors and assign real local archive images
            color_entries: list[dict[str, str]] = []
            seen_vi_colors: set[str] = set()
            images: list[dict[str, Any]] = []

            for img_idx, (_, r) in enumerate(rows.iterrows()):
                raw_col = str(r["colour_group_name"])
                vi_col, hex_code = COLOR_MAP.get(raw_col, ("Đen", "#1A1A1A"))
                local_img_url = matcher.pick_and_copy_image(
                    cat_name, target_gender, vi_col
                )

                images.append(
                    {
                        "url": local_img_url,
                        "altText": f"{vi_name} - Màu {vi_col} ({img_idx + 1})",
                        "sortOrder": img_idx,
                    }
                )
                if vi_col not in seen_vi_colors:
                    seen_vi_colors.add(vi_col)
                    color_entries.append(
                        {
                            "color": vi_col,
                            "colorHex": hex_code,
                            "imageUrl": local_img_url,
                        }
                    )

            # Ensure at least 2 colors per product for richer variant querying
            if len(color_entries) < 2:
                for fb_col, fb_hex in FALLBACK_COLORS:
                    if fb_col not in seen_vi_colors:
                        seen_vi_colors.add(fb_col)
                        fb_img_url = matcher.pick_and_copy_image(
                            cat_name, target_gender, fb_col
                        )
                        images.append(
                            {
                                "url": fb_img_url,
                                "altText": (
                                    f"{vi_name} - Màu {fb_col} ({len(images) + 1})"
                                ),
                                "sortOrder": len(images),
                            }
                        )
                        color_entries.append(
                            {
                                "color": fb_col,
                                "colorHex": fb_hex,
                                "imageUrl": fb_img_url,
                            }
                        )
                        break

            sizes = get_sizes_for_product(cat_name, target_gender)
            variants: list[dict[str, Any]] = []

            for c_idx, c_entry in enumerate(color_entries):
                for s_idx, size_label in enumerate(sizes):
                    sku = f"HM{p_code}-C{c_idx + 1}-{size_label}-{variant_seq}"
                    barcode_num = f"8936{variant_seq:09d}"
                    variant_seq += 1
                    is_oos = ((idx + c_idx * 3 + s_idx * 7) % 10) == 0
                    stock = 0 if is_oos else rng.randint(12, 95)

                    variants.append(
                        {
                            "sku": sku,
                            "barcode": barcode_num,
                            "size": size_label,
                            "color": c_entry["color"],
                            "colorHex": c_entry["colorHex"],
                            "imageUrl": c_entry["imageUrl"],
                            "listPrice": list_price,
                            "sellingPrice": selling_price,
                            "stock": stock,
                            "weight": rng.randint(180, 650),
                        }
                    )

            gender_vi = {
                "MALE": "nam giới",
                "FEMALE": "nữ giới",
                "UNISEX": "cả nam và nữ (Unisex)",
            }[target_gender]

            description = (
                f"{vi_name} (Mã H&M #{p_code}) dành cho {gender_vi}, "
                f"được chế tác từ chất liệu {profile['material']}. "
                f"Thiết kế phù hợp để {profile['occasion']}. "
                f"Ưu điểm phom dáng: {profile['body_fit']}. "
                f"Mô tả chi tiết gốc: {first_row['detail_desc']}"
            )

            brand_name = "H&M" if idx % 3 != 2 else BRANDS[idx % len(BRANDS)]

            selected_products.append(
                {
                    "hmProductCode": str(p_code),
                    "name": vi_name,
                    "slug": slug,
                    "categoryName": cat_name,
                    "brandName": brand_name,
                    "targetGender": target_gender,
                    "material": profile["material"],
                    "careInstructions": profile["care"],
                    "season": profile["season"],
                    "provider": "H&M Global Catalog",
                    "description": description,
                    "images": images,
                    "variants": variants,
                }
            )

    OUTPUT_JSON_PATH.write_text(
        json.dumps(selected_products, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Exported {len(selected_products)} enriched products to {OUTPUT_JSON_PATH}")
    print(
        f"Copied {len(matcher.copied_ids)} real .jpg images from "
        f"{ARCHIVE_IMAGES_DIR} to {BE_UPLOADS_DIR}"
    )


if __name__ == "__main__":
    main()
