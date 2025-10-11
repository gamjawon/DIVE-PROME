import 'package:frontend/domain/entities/location.dart';

abstract class CurrentLocationRepository {
  Future<Location> getCurrentLocation();
}
