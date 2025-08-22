import 'package:frontend/models/place_model.dart';

class SelectedPlace {
  final String name;
  final String address;
  final double latitude;
  final double longitude;

  SelectedPlace({
    required this.name,
    required this.address,
    required this.latitude,
    required this.longitude,
  });

  factory SelectedPlace.fromPlace(Place place) {
    return SelectedPlace(
      name: place.placeName,
      address: place.displayAddress,
      latitude: place.y,
      longitude: place.x,
    );
  }

  @override
  String toString() => name.isNotEmpty ? name : address;
}
