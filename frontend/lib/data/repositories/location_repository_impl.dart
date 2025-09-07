import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:frontend/data/datasources/location_datasource.dart';
import 'package:frontend/data/models/location_model.dart';
import 'package:riverpod_annotation/riverpod_annotation.dart';

part 'location_repository_impl.g.dart';

@Riverpod(keepAlive: true)
LocationRepositoryImpl locationRepository(Ref ref) {
  return LocationRepositoryImpl(ref.watch(locationDatasourceProvider));
}

class LocationRepositoryImpl {
  final LocationDatasource locationDatasource;

  LocationRepositoryImpl(this.locationDatasource);

  /// 현재 위치 불러오기
  Future<Location?> getCurrentLocation() {
    return locationDatasource.getCurrentLocation();
  }
}
