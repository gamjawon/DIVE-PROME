class Place {
  final String id;
  final String placeName;
  final String addressName;
  final String roadAddressName;
  final double x; // 경도
  final double y; // 위도
  final String categoryName;

  Place({
    required this.id,
    required this.placeName,
    required this.addressName,
    required this.roadAddressName,
    required this.x,
    required this.y,
    required this.categoryName,
  });

  factory Place.fromJson(Map<String, dynamic> json) {
    return Place(
      id: json['id'] ?? '',
      placeName: json['place_name'] ?? '',
      addressName: json['address_name'] ?? '',
      roadAddressName: json['road_address_name'] ?? '',
      x: double.parse(json['x'] ?? '0'),
      y: double.parse(json['y'] ?? '0'),
      categoryName: json['category_name'] ?? '',
    );
  }

  // 검색 결과 표시용 주소 (도로명 주소 우선, 없으면 지번 주소)
  String get displayAddress =>
      roadAddressName.isNotEmpty ? roadAddressName : addressName;
}

class PlaceSearchResponse {
  final List<Place> documents;
  final bool isEnd;

  PlaceSearchResponse({required this.documents, required this.isEnd});

  factory PlaceSearchResponse.fromJson(Map<String, dynamic> json) {
    return PlaceSearchResponse(
      documents:
          (json['documents'] as List<dynamic>?)
              ?.map((item) => Place.fromJson(item as Map<String, dynamic>))
              .toList() ??
          [],
      isEnd: json['meta']?['is_end'] ?? true,
    );
  }
}
