class Location {
  final double latitude;
  final double longitude;
  final String address;

  const Location({
    required this.latitude,
    required this.longitude,
    required this.address,
  });

  @override
  String toString() {
    return 'LocationModel(latitude: $latitude, longitude: $longitude, address: $address)';
  }
}
