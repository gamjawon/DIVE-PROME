import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:frontend/data/datasources/current_location_datasource.dart';
import 'package:frontend/data/models/location_model.dart';
import 'package:frontend/domain/entities/location.dart';
import 'package:frontend/domain/repositories/current_location_repository.dart';
import 'package:riverpod_annotation/riverpod_annotation.dart';

part 'current_location_repository_impl.g.dart';

@riverpod
CurrentLocationRepository currentLocationRepository(Ref ref) {
  return CurrentLocationRepositoryImpl(
    ref.watch(currentLocationDatasourceProvider),
  );
}

class CurrentLocationRepositoryImpl implements CurrentLocationRepository {
  final CurrentLocationDatasource currentLocationDatasource;

  CurrentLocationRepositoryImpl(this.currentLocationDatasource);

  @override
  Future<Location> getCurrentLocation() async {
    try {
      final currentLocation = await currentLocationDatasource
          .fetchCurrentLocation();
      return currentLocation.toEntity();
    } catch (e) {
      print('현재 위치 불러오는 중 에러 발생: $e');
      rethrow;
    }
  }
}
