"""Mẫu bài làm tham khảo Wikispeedia — Python 3.10 trở lên.

Tệp này cài đặt đầy đủ và chuẩn xác toàn bộ 10 hàm theo yêu cầu đề bài,
kèm chú thích chi tiết cho từng thuật toán để các thành viên tham khảo.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import numbers
from pathlib import Path


def load_graph(
    articles_path: Path, links_path: Path
) -> tuple[list[str], list[list[int]]]:
    """Đọc tập đỉnh và danh sách cạnh từ hai tệp TSV UTF-8.

    Bỏ dòng trống/dòng bắt đầu bằng #. Giữ ID mã hóa phần trăm.
    Trả (ids, out_neighbors): ids là danh sách ID phân biệt sắp theo byte
    UTF-8; out_neighbors[j] chứa các chỉ số đích của đỉnh ids[j].
    Gộp cạnh trùng, giữ cạnh tự nối và mọi đỉnh trong articles.tsv.
    Nút cụt có danh sách đích rỗng. Không gắn cứng số đỉnh/cạnh.
    Báo ValueError nếu đồ thị rỗng hoặc cạnh chứa ID ngoài tập đỉnh.
    """
    if not articles_path.is_file() or not links_path.is_file():
        raise ValueError("Tệp dữ liệu đồ thị không tồn tại")

    # 1. Đọc danh sách các đỉnh từ articles.tsv
    articles_text = articles_path.read_text(encoding="utf-8")
    raw_ids = set()
    for line in articles_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        raw_ids.add(line)

    if not raw_ids:
        raise ValueError("Tập đỉnh rỗng")

    # Sắp xếp các ID theo thứ tự từ điển của byte UTF-8
    ids = sorted(raw_ids, key=lambda s: s.encode("utf-8"))
    id_to_idx = {name: i for i, name in enumerate(ids)}
    n = len(ids)

    # 2. Đọc danh sách cạnh từ links.tsv
    links_text = links_path.read_text(encoding="utf-8")
    adj_sets: list[set[int]] = [set() for _ in range(n)]

    for line in links_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        src, dst = parts[0], parts[1]
        if src not in id_to_idx or dst not in id_to_idx:
            raise ValueError(f"Cạnh chứa ID ngoài tập đỉnh: {src} -> {dst}")
        u = id_to_idx[src]
        v = id_to_idx[dst]
        adj_sets[u].add(v)

    # Chuyển các set thành list đã sắp xếp cho tất định
    out_neighbors = [sorted(adj_sets[i]) for i in range(n)]
    return ids, out_neighbors


def pagerank(
    ids: list[str],
    out_neighbors: list[list[int]],
    beta: float,
    tol: float,
    max_iter: int,
    teleport: list[float] | None = None,
) -> list[float]:
    """Tính PageRank toàn cục hoặc theo phân phối dịch chuyển cho trước.

    ids và out_neighbors dùng cùng chỉ số như kết quả load_graph.
    beta thuộc (0, 1), tol > 0, max_iter là số nguyên >= 1.
    teleport=None dùng phân phối đều; nếu có, teleport phải dài len(ids),
    hữu hạn, không âm và có tổng lệch 1 không quá 1e-12.
    Khởi tạo bằng phân phối dịch chuyển; bù nút cụt luôn đều trên mọi đỉnh.
    Cập nhật đồng thời bằng số thực 64 bit. Dừng khi sai khác L1 giữa hai
    vòng <= tol hoặc đã thực hiện max_iter vòng; không chuẩn hóa lại.
    Trả vector mới nhất theo thứ tự ids, không trả thống kê phụ.
    Báo ValueError nếu tham số hoặc phân phối dịch chuyển không hợp lệ.
    """
    n = len(ids)
    if n == 0 or len(out_neighbors) != n:
        raise ValueError("Đồ thị rỗng hoặc không khớp kích thước")

    # Kiểm tra tính hợp lệ của tham số
    if isinstance(beta, bool) or not isinstance(beta, numbers.Real) or not (0.0 < beta < 1.0):
        raise ValueError("beta phải thuộc khoảng (0, 1)")
    if isinstance(tol, bool) or not isinstance(tol, numbers.Real) or tol <= 0.0 or not math.isfinite(tol):
        raise ValueError("tol phải là số thực dương hữu hạn")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter phải là số nguyên >= 1")

    # Xử lý vector teleport v (phân phối dịch chuyển ngẫu nhiên)
    if teleport is None:
        v = [1.0 / n] * n
    else:
        if len(teleport) != n:
            raise ValueError("Kích thước teleport không khớp số đỉnh")
        for x in teleport:
            if isinstance(x, bool) or not isinstance(x, numbers.Real) or not math.isfinite(x) or x < 0.0:
                raise ValueError("teleport chứa giá trị không hợp lệ (âm hoặc không hữu hạn)")
        if abs(math.fsum(teleport) - 1.0) > 1e-12:
            raise ValueError("Tổng xác suất teleport lệch 1 quá 1e-12")
        v = [float(x) for x in teleport]

    # Chuẩn bị danh sách cạnh đến (in-edges) và danh sách nút cụt (dangling nodes)
    incoming: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    dangling: list[int] = []
    for j, targets in enumerate(out_neighbors):
        d_j = len(targets)
        if d_j == 0:
            dangling.append(j)
        else:
            w = 1.0 / d_j
            for i in targets:
                incoming[i].append((j, w))

    # Khởi tạo vector điểm r^0 = v
    r = list(v)

    # Power Iteration
    one_over_n = 1.0 / n
    for _ in range(max_iter):
        # Tổng điểm ở các nút cụt: delta_t = sum_{j in dangling} r_j^t
        delta_t = math.fsum(r[j] for j in dangling)
        dangling_contrib = beta * delta_t * one_over_n

        new_r = [0.0] * n
        for i in range(n):
            inc_sum = math.fsum(r[j] * w for j, w in incoming[i])
            new_r[i] = beta * inc_sum + dangling_contrib + (1.0 - beta) * v[i]

        # Kiểm tra điều kiện dừng: chuẩn sai số L1 giữa 2 vòng
        diff = math.fsum(abs(new_r[i] - r[i]) for i in range(n))
        r = new_r
        if diff <= tol:
            break

    return r


def topic_distribution(
    ids: list[str], categories_path: Path, prefix: str
) -> list[float]:
    """Tạo phân phối dịch chuyển đều trên tập trang thuộc một chủ đề.

    Đọc categories.tsv UTF-8, bỏ dòng trống/dòng bắt đầu bằng #.
    Chọn nhãn bằng prefix hoặc bắt đầu bằng prefix + '.'. Một ID chỉ
    được đếm một lần trong tập chủ đề, dù có nhiều nhãn phù hợp.
    Trả vector dài len(ids), theo thứ tự ids: trang thuộc tập nhận 1/|S|,
    trang còn lại nhận 0. Báo ValueError nếu tập chủ đề rỗng hoặc dữ liệu
    nhãn tham chiếu ID ngoài ids. Không dùng nội dung bài để suy ra nhãn.
    """
    if not categories_path.is_file():
        raise ValueError("Tệp categories.tsv không tồn tại")

    id_set = set(ids)
    selected = set()
    dot_prefix = prefix + "."

    categories_text = categories_path.read_text(encoding="utf-8")
    for line in categories_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        art, label = parts[0], parts[1]
        if art not in id_set:
            raise ValueError(f"Dữ liệu nhãn tham chiếu ID ngoài ids: {art}")
        if label == prefix or label.startswith(dot_prefix):
            selected.add(art)

    if not selected:
        raise ValueError(f"Tập chủ đề rỗng cho tiền tố {prefix}")

    prob = 1.0 / len(selected)
    return [prob if art in selected else 0.0 for art in ids]


def mix_pageranks(
    topic_scores: list[list[float]], weights: list[float]
) -> list[float]:
    """Ghép các vector PageRank bằng tổng có trọng số theo từng tọa độ.

    topic_scores gồm các vector cùng độ dài, cùng thứ tự đỉnh.
    weights có một trọng số cho mỗi vector, hữu hạn, không âm và có
    tổng lệch 1 không quá 1e-12. Trả một vector có cùng độ dài.
    Cấu hình chính thức dùng thứ tự Science, History, Geography và trọng
    số 0.5, 0.3, 0.2. Không chạy lại PageRank hoặc làm tròn điểm ở đây.
    Báo ValueError nếu danh sách rỗng, kích thước hoặc trọng số không hợp lệ.
    """
    if not topic_scores or not weights:
        raise ValueError("Danh sách vector hoặc trọng số rỗng")
    if len(topic_scores) != len(weights):
        raise ValueError("Số vector và số trọng số không khớp nhau")

    n = len(topic_scores[0])
    if n == 0:
        raise ValueError("Vector điểm có độ dài 0")

    for vec in topic_scores:
        if len(vec) != n:
            raise ValueError("Các vector điểm không cùng độ dài")

    for w in weights:
        if isinstance(w, bool) or not isinstance(w, numbers.Real) or not math.isfinite(w) or w < 0.0:
            raise ValueError("Trọng số phải là số thực không âm và hữu hạn")

    if abs(math.fsum(weights) - 1.0) > 1e-12:
        raise ValueError("Tổng trọng số lệch 1 quá 1e-12")

    m = len(weights)
    result = [0.0] * n
    for i in range(n):
        result[i] = math.fsum(weights[k] * topic_scores[k][i] for k in range(m))

    return result


def normalize_text(text: str) -> str:
    r"""Chuẩn hóa chuỗi Unicode theo đúng quy tắc của phần C.

    Chỉ đổi ASCII A–Z thành a–z. Gộp mỗi dãy ký tự thuộc tập dấu cách
    ASCII, \t, \n, \r, \v, \f thành một dấu cách ASCII; bỏ dấu cách ASCII
    ở hai đầu. Giữ nguyên các ký tự khác, dấu câu và toàn bộ nội dung.
    Không chuẩn hóa Unicode, bỏ dấu hoặc gọi lower() cho toàn bộ Unicode.
    Trả chuỗi đã chuẩn hóa; chuỗi rỗng hoặc chỉ có khoảng trắng trên trả ''.
    """
    result: list[str] = []
    space_pending = False
    whitespace_set = {" ", "\t", "\n", "\r", "\v", "\f"}

    for ch in text:
        if ch in whitespace_set:
            if result:
                space_pending = True
        else:
            if space_pending:
                result.append(" ")
                space_pending = False
            # Chỉ chuyển đổi ký tự ASCII A-Z sang a-z
            if "A" <= ch <= "Z":
                result.append(chr(ord(ch) + 32))
            else:
                result.append(ch)

    return "".join(result)


def shingles(text: str, k: int) -> set[str]:
    """Tạo tập k-shingle từ chuỗi ĐÃ được normalize_text xử lý.

    Mỗi shingle là k code point Unicode liên tiếp, không phải k byte/từ.
    Trả tập chuỗi phân biệt; không lưu số lần xuất hiện hoặc vị trí.
    Chuỗi ngắn hơn k trả tập rỗng. Báo ValueError nếu k không phải số
    nguyên dương. Không chuẩn hóa văn bản thêm lần nữa trong hàm này.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k phải là số nguyên dương")

    n = len(text)
    if n < k:
        return set()

    return {text[i : i + k] for i in range(n - k + 1)}


def shingle_id(shingle: str, p: int) -> int:
    """Ánh xạ một shingle thành số nguyên trong [0, p-1].

    Băm byte UTF-8 bằng SHA-256, lấy 4 byte đầu, đọc số nguyên không dấu
    big-endian rồi lấy modulo p. Không dùng hash() mặc định của Python.
    Trả int tất định giữa các lần chạy. Báo ValueError nếu p <= 1.
    """
    if isinstance(p, bool) or not isinstance(p, int) or p <= 1:
        raise ValueError("p phải là số nguyên > 1")

    digest = hashlib.sha256(shingle.encode("utf-8")).digest()
    val = int.from_bytes(digest[:4], "big")
    return val % p


def minhash(shingle_ids: set[int], hash_params: dict) -> list[int]:
    """Tính chữ ký từ tập mã shingle và bộ hàm băm dùng chung.

    hash_params có p và hash_functions là danh sách {'a': int, 'b': int}
    theo thứ tự tọa độ. Mỗi tọa độ là min((a*x+b) % p) trên shingle_ids.
    Tập rỗng trả toàn giá trị p. Trả list[int] dài bằng số hàm băm;
    không gắn cứng 128 vì bộ chấm có thể dùng bộ hệ số nhỏ khác.
    Mã đầu vào thuộc [0, p-1]; phép nhân/cộng/modulo dùng số nguyên chính
    xác. Không sinh hệ số mới, không trả ID phần tử đạt min.
    """
    p: int = hash_params["p"]
    funcs: list[dict[str, int]] = hash_params["hash_functions"]

    if not shingle_ids:
        return [p] * len(funcs)

    sig: list[int] = []
    for hf in funcs:
        a = hf["a"]
        b = hf["b"]
        m = min((a * x + b) % p for x in shingle_ids)
        sig.append(m)

    return sig


def jaccard(set_a: set[str], set_b: set[str]) -> float:
    """Tính Jaccard chính xác trên hai tập CHUỖI shingle trước bước băm.

    Trả kích thước giao chia kích thước hợp. Theo quy ước bài tập, hai tập
    rỗng trả 1.0; chỉ một tập rỗng trả 0.0. Không dùng tập mã băm thay thế.
    """
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0

    inter_size = len(set_a & set_b)
    union_size = len(set_a) + len(set_b) - inter_size
    return inter_size / union_size


def signature_similarity(
    sig_a: list[int], sig_b: list[int], empty_a: bool, empty_b: bool
) -> float:
    """Ước lượng Jaccard bằng tỷ lệ tọa độ chữ ký trùng nhau.

    Hai chữ ký phải cùng độ dài dương, dùng cùng bộ hàm băm và thứ tự.
    empty_a/empty_b cho biết tập shingle gốc có rỗng hay không.
    Hai tập rỗng trả 1.0; chỉ một tập rỗng trả 0.0. Các trường hợp còn
    lại trả số tọa độ bằng nhau chia độ dài chữ ký. Không chuyển chữ ký
    thành tập. Báo ValueError nếu độ dài không hợp lệ.
    """
    if len(sig_a) != len(sig_b) or len(sig_a) == 0:
        raise ValueError("Độ dài chữ ký không hợp lệ hoặc không khớp nhau")

    if empty_a and empty_b:
        return 1.0
    if empty_a or empty_b:
        return 0.0

    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def main() -> None:
    """Đọc tham số, gọi các hàm và ghi bốn tệp kết quả CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--hash-params", type=Path, default=Path("assignment/minhash_params.json")
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("data/pairs.csv"),
        help="CSV có header article_id_a,article_id_b; mặc định dùng bộ cặp Wikispeedia chính thức.",
    )
    parser.add_argument("--beta", type=float, default=0.85)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--max-iter", type=int, default=1000)
    args = parser.parse_args()

    graph_dir = args.data_dir / "wikispeedia_paths-and-graph"
    text_dir = args.data_dir / "plaintext_articles"
    with args.hash_params.open(encoding="utf-8") as handle:
        hash_params = json.load(handle)
    with args.pairs.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["article_id_a", "article_id_b"]:
            raise ValueError("Tệp cặp cần header article_id_a,article_id_b")
        pairs = [(row["article_id_a"], row["article_id_b"]) for row in reader]

    ids, out_neighbors = load_graph(
        graph_dir / "articles.tsv", graph_dir / "links.tsv"
    )
    required_ids = {article for pair in pairs for article in pair}
    if not required_ids.issubset(set(ids)):
        raise ValueError("Tệp cặp chứa ID không có trong articles.tsv")

    # A. Điểm PageRank toàn cục.
    scores = pagerank(ids, out_neighbors, args.beta, args.tol, args.max_iter)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "pagerank.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["article_id", "score"])
        for i, article in enumerate(ids):
            writer.writerow([article, format(scores[i], ".17g")])

    # B. Ba chủ đề và r_mix trong cùng một tệp.
    topics = ["Science", "History", "Geography"]
    topic_scores = []
    for topic in topics:
        teleport = topic_distribution(
            ids, graph_dir / "categories.tsv", "subject." + topic
        )
        topic_scores.append(
            pagerank(ids, out_neighbors, args.beta, args.tol, args.max_iter, teleport)
        )
    mixed_scores = mix_pageranks(topic_scores, [0.5, 0.3, 0.2])
    with (args.output_dir / "topic_pagerank.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["topic", "article_id", "score"])
        for topic, vector in zip(topics + ["r_mix"], topic_scores + [mixed_scores]):
            for i, article in enumerate(ids):
                writer.writerow([topic, article, format(vector[i], ".17g")])

    # C. Chữ ký cho mọi bài; chỉ giữ tập chuỗi cho các cặp cần so sánh.
    signatures = {}
    pair_shingles = {}
    with (args.output_dir / "minhash.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        length = len(hash_params["hash_functions"])
        writer.writerow(
            ["article_id", "shingle_count"] + [f"h{j:03d}" for j in range(length)]
        )
        for article in ids:
            text = (text_dir / (article + ".txt")).read_text(encoding="utf-8")
            parts = shingles(normalize_text(text), hash_params["shingle_length"])
            codes = {shingle_id(part, hash_params["p"]) for part in parts}
            signature = minhash(codes, hash_params)
            signatures[article] = signature
            if article in required_ids:
                pair_shingles[article] = parts
            writer.writerow([article, len(parts), *signature])

    with (args.output_dir / "similarities.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["article_id_a", "article_id_b", "jaccard_exact", "jaccard_estimate"]
        )
        for article_a, article_b in pairs:
            set_a, set_b = pair_shingles[article_a], pair_shingles[article_b]
            exact = jaccard(set_a, set_b)
            estimate = signature_similarity(
                signatures[article_a], signatures[article_b], not set_a, not set_b
            )
            writer.writerow(
                [article_a, article_b, format(exact, ".17g"), format(estimate, ".17g")]
            )

    print(f"Đã ghi bốn tệp kết quả vào {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

