import 'package:frontend/domain/entities/location.dart';
import 'package:frontend/domain/value_objects/search_result.dart';

abstract class PlaceSearchRepository {
  Future<SearchResult<Location>> getPlaces(String query);
}
